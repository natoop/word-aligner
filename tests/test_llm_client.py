from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.llm_client import ChatCompletionResult, LLMProviderError, OpenAIChatClient
from app.llm_config import LLMProvider, LLMSettings

_SCHEMA = {"type": "object", "properties": {"links": {"type": "array"}}, "required": ["links"]}
_MESSAGES = [{"role": "user", "content": "align these tokens"}]


def _provider(*, response_format: str = "json_object", api_key: str = "secret-key") -> LLMProvider:
    return LLMProvider(
        id="test",
        name="Test provider",
        base_url="https://llm.example.test/v1/",
        model="test-model",
        api_key=api_key,
        response_format=response_format,
    )


def _settings(*, max_output_tokens: int = 1234, timeout_seconds: float = 5) -> LLMSettings:
    return LLMSettings(max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds)


def _success_payload(
    *, content: str = '{"links":[]}', finish_reason: str = "stop", usage: dict | None = None, **message_values
) -> dict:
    message = {"role": "assistant", "content": content, **message_values}
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
    }


def _run(coro):
    return asyncio.run(coro)


def test_complete_posts_openai_compatible_json_schema_request_and_returns_only_content_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_payload(), request=request)

    result = _run(
        OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(
            _provider(response_format="json_schema"), _MESSAGES, _SCHEMA
        )
    )

    request = seen[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert str(request.url) == "https://llm.example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    assert body == {
        "model": "test-model",
        "messages": _MESSAGES,
        "stream": False,
        "max_tokens": 1234,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "translation_alignment", "strict": True, "schema": _SCHEMA},
        },
    }
    assert "temperature" not in body
    assert result == ChatCompletionResult(
        content='{"links":[]}', usage={"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}
    )


def test_client_uses_configured_completion_token_parameter() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_success_payload(), request=request)

    provider = LLMProvider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        max_tokens_parameter="max_completion_tokens",
    )
    _run(OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(provider, _MESSAGES, _SCHEMA))

    assert seen[0]["max_completion_tokens"] == 1234
    assert "max_tokens" not in seen[0]


@pytest.mark.parametrize(
    ("response_format", "expected_format"),
    [("json_object", {"type": "json_object"}), ("none", None)],
)
def test_response_format_configuration_controls_request_body(
    response_format: str, expected_format: dict | None
) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_success_payload(), request=request)

    _run(
        OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(
            _provider(response_format=response_format), _MESSAGES, _SCHEMA
        )
    )

    assert seen[0].get("response_format") == expected_format
    if expected_format is None:
        assert "response_format" not in seen[0]


def test_provider_switching_and_empty_api_key_omit_authorization_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_payload(), request=request)

    async def complete_both() -> None:
        client = OpenAIChatClient(_settings(), httpx.MockTransport(handler))
        await client.complete(_provider(api_key=""), _MESSAGES, _SCHEMA)
        second_provider = LLMProvider(
            id="other", name="Other", base_url="http://localhost:8000/v1", model="local-model"
        )
        await client.complete(second_provider, _MESSAGES, _SCHEMA)

    _run(complete_both())

    assert [str(request.url) for request in seen] == [
        "https://llm.example.test/v1/chat/completions",
        "http://localhost:8000/v1/chat/completions",
    ]
    assert all("authorization" not in request.headers for request in seen)
    assert json.loads(seen[1].content)["model"] == "local-model"


@pytest.mark.parametrize(
    ("status", "code", "service_status", "marker"),
    [
        (401, "provider_auth_error", 502, "sensitive-upstream-body"),
        (403, "provider_auth_error", 502, "sensitive-upstream-body"),
        (429, "provider_rate_limited", 429, "sensitive-upstream-body"),
        (503, "provider_http_error", 502, "sensitive-upstream-body"),
    ],
)
def test_http_errors_map_to_safe_messages(status: int, code: str, service_status: int, marker: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=marker, headers={"x-private": marker}, request=request)

    with pytest.raises(LLMProviderError) as error:
        _run(OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(_provider(), _MESSAGES, _SCHEMA))

    assert error.value.code == code
    assert error.value.status_code == service_status
    assert marker not in str(error.value)
    assert "llm.example.test" not in str(error.value)
    assert "secret-key" not in str(error.value)


@pytest.mark.parametrize(
    ("exception_type", "code", "status"),
    [
        (httpx.ReadTimeout, "provider_timeout", 504),
        (httpx.ConnectError, "provider_network_error", 502),
    ],
)
def test_transport_errors_map_without_forwarding_exception_details(exception_type, code: str, status: int) -> None:
    marker = "private-network-details"

    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_type(marker, request=request)

    with pytest.raises(LLMProviderError) as error:
        _run(OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(_provider(), _MESSAGES, _SCHEMA))

    assert error.value.code == code
    assert error.value.status_code == status
    assert marker not in str(error.value)
    assert "llm.example.test" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-provider-secret",
        b"[]",
        b"{}",
        b'{"choices": []}',
        b'{"choices": [{"message": [], "finish_reason": "stop"}]}',
    ],
)
def test_malformed_provider_body_returns_safe_error(payload: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, request=request)

    with pytest.raises(LLMProviderError, match="invalid response") as error:
        _run(OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(_provider(), _MESSAGES, _SCHEMA))

    assert "provider-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("payload", "code", "message"),
    [
        (_success_payload(finish_reason="length"), "output_limit_reached", "output limit"),
        (_success_payload(refusal="private refusal text"), "provider_refused", "allowed response"),
        (_success_payload(finish_reason="refusal"), "provider_refused", "allowed response"),
        (_success_payload(content="  "), "empty_provider_response", "no content"),
        (_success_payload(finish_reason="content_filter"), "provider_refused", "allowed response"),
    ],
)
def test_refusal_truncation_and_empty_content_are_safe_errors(payload: dict, code: str, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(LLMProviderError, match=message) as error:
        _run(OpenAIChatClient(_settings(), httpx.MockTransport(handler)).complete(_provider(), _MESSAGES, _SCHEMA))

    assert error.value.code == code
    assert "private refusal text" not in str(error.value)


def test_missing_usage_returns_none_and_negative_or_non_integer_usage_is_rejected() -> None:
    def missing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": _success_payload()["choices"]}, request=request)

    result = _run(
        OpenAIChatClient(_settings(), httpx.MockTransport(missing_handler)).complete(_provider(), _MESSAGES, _SCHEMA)
    )
    assert result.usage is None

    def invalid_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_payload(usage={"total_tokens": -1}), request=request)

    with pytest.raises(LLMProviderError, match="invalid usage") as error:
        _run(
            OpenAIChatClient(_settings(), httpx.MockTransport(invalid_handler)).complete(
                _provider(), _MESSAGES, _SCHEMA
            )
        )
    assert error.value.code == "invalid_provider_response"
