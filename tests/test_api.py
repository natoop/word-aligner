from __future__ import annotations

import asyncio
import re

import httpx
import numpy as np

from app.alignment import AlignmentService, TokenEmbeddingAlignment
from app.config import Settings
from app.main import create_app
from app.tokenization import WordTokenizer


def _whitespace_tokenize(value: str):
    for match in re.finditer(r"\S+", value):
        yield match.group(), match.start(), match.end()


class StubBackend:
    def align_tokens(self, source_tokens: list[str], target_tokens: list[str]) -> TokenEmbeddingAlignment:
        similarities = np.asarray([[0.9, 0.8]], dtype=float)
        source_probabilities = _softmax(similarities, axis=1)
        target_probabilities = _softmax(similarities, axis=0)
        return TokenEmbeddingAlignment(
            alignments={"itermax": {(0, 0), (0, 1)}},
            similarities=_freeze(similarities),
            source_to_target_probabilities=_freeze(source_probabilities),
            target_to_source_probabilities=_freeze(target_probabilities),
        )


def _softmax(matrix: np.ndarray, *, axis: int) -> np.ndarray:
    scaled = matrix / 0.1
    scaled -= scaled.max(axis=axis, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _freeze(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _app():
    settings = Settings(eager_load=False)
    service = AlignmentService(
        settings,
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=StubBackend,
    )
    return create_app(settings, service)


async def _request(method: str, path: str, json: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=json)


def test_align_endpoint_returns_languages_and_grouped_links() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/api/v1/align",
            json={
                "source_language": "en",
                "target_language": "zh-Hans",
                "sentence_pairs": [
                    {
                        "id": "sentence-1",
                        "source": "hello",
                        "target": "你 好",
                    }
                ],
            },
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_language"] == "en"
    assert body["target_language"] == "zh-Hans"
    assert body["embedding_layer"] == 8
    assert body["confidence_method"] == "bidirectional-margin-span-v2"
    assert body["sentence_alignments"][0]["alignment_groups"][0]["type"] == "one-to-many"
    assert body["sentence_alignments"][0]["alignment_groups"][0]["target_indices"] == [0, 1]
    assert body["sentence_alignments"][0]["alignment_groups"][0]["origin"] == "model"
    assert 0.0 <= body["sentence_alignments"][0]["alignment_groups"][0]["similarity"] <= 1.0
    assert 0.0 <= body["sentence_alignments"][0]["alignment_groups"][0]["confidence"] <= 1.0
    links = body["sentence_alignments"][0]["links"]
    assert all(0.0 <= link["similarity"] <= 1.0 for link in links)
    assert all(0.0 <= link["confidence"] <= 1.0 for link in links)


def test_align_endpoint_rejects_an_invalid_language_code() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/api/v1/align",
            json={
                "source_language": "english!",
                "target_language": "zh-Hans",
                "sentence_pairs": [{"source": "hello", "target": "你好"}],
            },
        )
    )

    assert response.status_code == 422


def test_align_endpoint_rejects_an_invalid_repair_distance() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/api/v1/align",
            json={
                "source_language": "en",
                "target_language": "zh-Hans",
                "repair": {"max_position_distance": 1.1},
                "sentence_pairs": [{"source": "hello", "target": "你好"}],
            },
        )
    )

    assert response.status_code == 422


def test_align_endpoint_rejects_an_invalid_repair_confidence() -> None:
    response = asyncio.run(
        _request(
            "POST",
            "/api/v1/align",
            json={
                "source_language": "en",
                "target_language": "zh-Hans",
                "repair": {"min_confidence": -0.1},
                "sentence_pairs": [{"source": "hello", "target": "你好"}],
            },
        )
    )

    assert response.status_code == 422


def test_health_endpoints_do_not_load_the_model_in_lazy_mode() -> None:
    async def get_health_responses() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health/live"), await client.get("/health/ready")

    live, ready = asyncio.run(get_health_responses())

    assert live.json() == {"status": "ok"}
    assert ready.json()["status"] == "ready"
    assert ready.json()["model_loaded"] is False
    assert ready.json()["load_mode"] == "lazy"


def test_languages_endpoint_returns_supported_languages_without_loading_model() -> None:
    response = asyncio.run(_request("GET", "/api/v1/languages"))

    assert response.status_code == 200
    body = response.json()
    languages = {language["code"]: language for language in body["languages"]}
    assert body["model"] == "xlmr"
    assert body["pairing"] == "any-to-any"
    assert body["total"] == len(body["languages"])
    assert languages["en"]["name"] == "English"
    assert languages["zh-Hans"]["tokenizer"] == "jieba"
