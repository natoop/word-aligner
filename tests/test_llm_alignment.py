from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import regex

from app.config import Settings
from app.llm_alignment import LLMAlignmentService, tokenize_for_llm
from app.llm_client import OpenAIChatClient
from app.llm_config import LLMProvider, LLMSettings
from app.main import create_app


def _provider(provider_id: str = "alpha") -> LLMProvider:
    return LLMProvider(
        id=provider_id,
        name=f"{provider_id} provider",
        base_url=f"https://{provider_id}.example/v1",
        model=f"{provider_id}-model",
        api_key=f"secret-{provider_id}",
    )


def _app(handler, *, providers: tuple[LLMProvider, ...] | None = None):
    settings = LLMSettings(providers=providers if providers is not None else (_provider(),))
    service = LLMAlignmentService(settings, OpenAIChatClient(settings, transport=httpx.MockTransport(handler)))
    return create_app(Settings(), llm_alignment_service=service)


def _request(app, path: str = "/api/v1/llm/align", *, payload: dict | None = None, method: str = "POST"):
    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(request())


def _input(source: str = "hello", target: str = "你好", **extra) -> dict:
    return {
        "source_language": "en",
        "target_language": "zh-Hans",
        "sentence_pairs": [{"id": "pair-1", "source": source, "target": target}],
        **extra,
    }


def _completion(content: dict | str, usage: dict | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content if isinstance(content, str) else json.dumps(content)},
                }
            ],
            "usage": usage,
        },
    )


def _aligned(source_ids: list[str], target_ids: list[str], **extra) -> dict:
    return {
        "alignments": [{"source_ids": source_ids, "target_ids": target_ids}],
        "source_unaligned": [],
        "target_unaligned": [],
        **extra,
    }


def test_llm_alignment_preserves_original_utf16_offsets_and_emoji_graphemes() -> None:
    source = "  red  blue\nred 👩🏽‍💻 cafe\u0301  "
    target = "red red blue 👩🏽‍💻 café"

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        assert data["source"] == source
        assert data["target"] == target
        assert data["source_units"][3] == {"id": "s3", "text": "👩🏽‍💻"}
        return _completion(
            {
                "alignments": [
                    {"source_ids": [f"s{source_index}"], "target_ids": [f"t{target_index}"]}
                    for source_index, target_index in [(0, 1), (1, 2), (2, 0), (3, 3), (4, 4)]
                ],
                "source_unaligned": [],
                "target_unaligned": [],
            },
            {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        )

    app = _app(handler)
    response = _request(app, payload=_input(source, target))
    assert response.status_code == 200
    body = response.json()
    assert body["offset_unit"] == "utf16"
    assert body["method"] == "llm"
    assert body["provider"] == "alpha"
    assert body["confidence_method"] is None
    assert body["usage"]["total_tokens"] == 80
    sentence = body["sentence_alignments"][0]
    for side, original in (("source", source), ("target", target)):
        assert sentence[side] == original
        for token in sentence[f"{side}_tokens"]:
            encoded = original.encode("utf-16-le")
            assert encoded[token["start"] * 2 : token["end"] * 2].decode("utf-16-le") == token["text"]
    emoji = sentence["source_tokens"][3]
    assert emoji["end"] - emoji["start"] == 7
    assert sentence["source_tokens"][4]["text"] == "cafe\u0301"
    assert [group["target_indices"] for group in sentence["alignment_groups"][:3]] == [[1], [2], [0]]
    assert all(group["confidence"] is None and group["similarity"] is None for group in sentence["alignment_groups"])
    assert sentence["links"] == []
    assert app.state.alignment_service.model_loaded is False


def test_llm_alignment_supports_noncontiguous_phrases_and_explicit_unaligned_units() -> None:
    result = {
        "alignments": [
            {"source_ids": ["s2", "s0"], "target_ids": ["t4", "t2", "t3"]},
            {"source_ids": ["s1"], "target_ids": ["t1"]},
        ],
        "source_unaligned": [],
        "target_unaligned": ["t0"],
    }
    response = _request(_app(lambda _: _completion(result)), payload=_input("pick it up", "把它捡起来"))
    assert response.status_code == 200
    sentence = response.json()["sentence_alignments"][0]
    group = sentence["alignment_groups"][0]
    assert group["source_indices"] == [0, 2]
    assert group["target_indices"] == [2, 3, 4]
    assert group["type"] == "many-to-many"
    assert group["origin"] == "llm"
    assert group["links"] == []
    assert sentence["unaligned_target_indices"] == [0]


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "[]",
        {"alignments": []},
        _aligned(["s99"], ["t0"]),
        _aligned(["s0", "s0"], ["t0"]),
        _aligned(["t0"], ["t0"]),
        _aligned([], ["t0"]),
        _aligned([0], ["t0"]),
        _aligned(["s0"], []),
        _aligned(["s0"], ["t0"], confidence=1),
        _aligned(["s0"], ["t0"], source_unaligned=["s0"]),
        {"alignments": [], "source_unaligned": [], "target_unaligned": ["t0"]},
        {
            "alignments": [
                {"source_ids": ["s0"], "target_ids": ["t0"]},
                {"source_ids": ["s0"], "target_ids": ["t0"]},
            ],
            "source_unaligned": [],
            "target_unaligned": [],
        },
        '{"alignments":[],"alignments":[],"source_unaligned":["s0"],"target_unaligned":["t0"]}',
    ],
)
def test_llm_alignment_rejects_invalid_or_incomplete_model_ids(content) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return _completion(content)

    response = _request(_app(handler), payload=_input("hello", "bonjour"))
    assert response.status_code == 502
    assert response.json()["error"] == "llm_invalid_alignment"
    assert len(calls) == 1  # Invalid output never silently triggers additional billable requests.


def test_llm_alignment_accepts_all_units_explicitly_unaligned() -> None:
    response = _request(
        _app(lambda _: _completion({"alignments": [], "source_unaligned": ["s0"], "target_unaligned": ["t0"]})),
        payload=_input("hello", "bonjour"),
    )
    assert response.status_code == 200
    sentence = response.json()["sentence_alignments"][0]
    assert sentence["alignment_groups"] == []
    assert sentence["unaligned_source_indices"] == [0]
    assert sentence["unaligned_target_indices"] == [0]


def test_llm_provider_selection_and_catalog_never_expose_credentials_or_urls() -> None:
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        return _completion(_aligned(["s0"], ["t0"]))

    app = _app(handler, providers=(_provider("alpha"), _provider("beta")))
    catalog = _request(app, "/api/v1/llm/providers", method="GET")
    assert catalog.status_code == 200
    assert catalog.json()["default_provider"] == "alpha"
    assert [item["id"] for item in catalog.json()["providers"]] == ["alpha", "beta"]
    assert "secret" not in catalog.text
    assert "https://" not in catalog.text
    for provider in (None, "beta"):
        payload = _input("hello", "bonjour")
        if provider:
            payload["provider"] = provider
        response = _request(app, payload=payload)
        assert response.status_code == 200
        assert response.json()["provider"] == (provider or "alpha")
    assert [str(request.url) for request in calls] == [
        "https://alpha.example/v1/chat/completions",
        "https://beta.example/v1/chat/completions",
    ]
    assert calls[1].headers["Authorization"] == "Bearer secret-beta"


def test_llm_missing_or_unknown_provider_does_not_call_upstream() -> None:
    def handler(_):
        pytest.fail("Must not call the provider")

    unconfigured = _app(handler, providers=())
    catalog = _request(unconfigured, "/api/v1/llm/providers", method="GET")
    assert catalog.json() == {"providers": [], "default_provider": None}
    assert _request(unconfigured, payload=_input()).status_code == 503
    assert _request(_app(handler), payload=_input(provider="unknown")).status_code == 400


def test_llm_validates_every_pair_before_any_upstream_request() -> None:
    def handler(_):
        pytest.fail("Oversized later pairs must be rejected before any paid requests")

    payload = _input()
    payload["sentence_pairs"].append({"source": "word " * 1025, "target": "bonjour"})
    response = _request(_app(handler), payload=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "llm_input_too_large"


@pytest.mark.parametrize(
    "changes",
    [
        {"sentence_pairs": []},
        {"sentence_pairs": [{"source": "a", "target": "b"}] * 11},
        {"sentence_pairs": [{"source": " \n ", "target": "b"}]},
        {"sentence_pairs": [{"source": "a" * 10001, "target": "b"}]},
        {"method": "itermax"},
        {"base_url": "https://untrusted.example/v1"},
        {"source_language": "invalid!"},
    ],
)
def test_llm_invalid_input_is_rejected_before_calling_provider(changes) -> None:
    def handler(_):
        pytest.fail("Invalid input must not call upstream")

    response = _request(_app(handler), payload=_input(**changes))
    assert response.status_code == 422


def test_llm_batch_usage_is_provider_reported_and_missing_usage_is_not_fabricated() -> None:
    usages = [{"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}, None]

    def handler(_):
        return _completion(_aligned(["s0"], ["t0"]), usages.pop(0))

    payload = _input("hello", "bonjour")
    payload["sentence_pairs"] *= 2
    response = _request(_app(handler), payload=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["usage"] is None
    assert body["sentence_alignments"][0]["usage"]["total_tokens"] == 6
    assert body["sentence_alignments"][1]["usage"] is None

    app = _app(lambda _: _completion(_aligned(["s0"], ["t0"]), {"total_tokens": 6}))
    summed = _request(app, payload=payload)
    assert summed.json()["usage"] == {"total_tokens": 12}


def test_llm_routes_have_openapi_contracts_and_safe_errors() -> None:
    app = _app(lambda _: httpx.Response(401, text="secret-alpha credential echoed by upstream"))
    document = app.openapi()
    assert "/api/v1/llm/align" in document["paths"]
    assert "/api/v1/llm/providers" in document["paths"]
    schema = document["components"]["schemas"]["LLMAlignmentResponse"]
    assert schema["properties"]["offset_unit"]["const"] == "utf16"
    assert "api_key" not in document["components"]["schemas"]["LLMProviderInfo"]["properties"]
    response = _request(app, payload=_input())
    assert response.status_code == 502
    assert "secret-alpha" not in response.text
    assert "alpha.example" not in response.text


def test_llm_tokenizer_keeps_arabic_combining_marks_and_placeholders() -> None:
    text = "مرحبًا [[name]] ${user} 中文"
    tokens = tokenize_for_llm(text)
    assert [token.text for token in tokens] == ["مرحبًا", "[[name]]", "${user}", "中", "文"]
    for token in tokens:
        assert text.encode("utf-16-le")[2 * token.start : 2 * token.end].decode("utf-16-le") == token.text


def test_llm_tokenizer_never_splits_an_indic_conjunct_or_emoji_grapheme() -> None:
    text = "क्\u200dष 👨‍👩‍👧‍👦 a\u200c café\u0301"
    boundaries = {0}
    for match in regex.finditer(r"\X", text):
        boundaries.add(len(text[: match.end()].encode("utf-16-le")) // 2)
    for token in tokenize_for_llm(text):
        assert token.start in boundaries
        assert token.end in boundaries
