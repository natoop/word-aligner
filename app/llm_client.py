from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.llm_config import LLMProvider, LLMSettings


class LLMProviderError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    content: str
    usage: dict[str, int] | None


class OpenAIChatClient:
    def __init__(self, settings: LLMSettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    async def complete(
        self,
        provider: LLMProvider,
        messages: list[dict[str, str]],
        response_schema: dict,
    ) -> ChatCompletionResult:
        if not isinstance(provider, LLMProvider):
            raise LLMProviderError("invalid_request", "Invalid LLM provider configuration.", status_code=400)
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in message.items())
            for message in messages
        ):
            raise LLMProviderError("invalid_request", "Chat messages must be a list of string-valued objects.", 400)
        if not isinstance(response_schema, dict):
            raise LLMProviderError("invalid_request", "The response schema must be an object.", 400)

        body: dict[str, Any] = {
            "model": provider.model,
            "messages": messages,
            "stream": False,
            provider.max_tokens_parameter: self._settings.max_output_tokens,
        }
        if provider.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif provider.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_alignment",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        try:
            json.dumps(body, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            raise LLMProviderError("invalid_request", "The chat completion request is invalid.", 400) from None

        headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else None
        url = f"{provider.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self._settings.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise LLMProviderError("provider_timeout", "The LLM provider request timed out.", 504) from None
        except httpx.RequestError:
            raise LLMProviderError("provider_network_error", "The LLM provider could not be reached.", 502) from None
        except (TypeError, ValueError):
            raise LLMProviderError("invalid_request", "The chat completion request is invalid.", 400) from None

        if response.status_code == 408:
            raise LLMProviderError("provider_timeout", "The LLM provider request timed out.", 504)
        if response.status_code in {401, 403}:
            raise LLMProviderError("provider_auth_error", "The LLM provider rejected the configured credentials.", 502)
        if response.status_code == 429:
            raise LLMProviderError("provider_rate_limited", "The LLM provider is rate limiting requests.", 429)
        if not 200 <= response.status_code < 300:
            raise LLMProviderError("provider_http_error", "The LLM provider returned an unsuccessful response.", 502)

        try:
            data = response.json()
        except (ValueError, UnicodeDecodeError):
            raise LLMProviderError(
                "invalid_provider_response", "The LLM provider returned an invalid response."
            ) from None

        if not isinstance(data, dict):
            raise LLMProviderError("invalid_provider_response", "The LLM provider returned an invalid response.")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProviderError("invalid_provider_response", "The LLM provider returned an invalid response.")

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise LLMProviderError("output_limit_reached", "The LLM provider response exceeded the output limit.")
        if finish_reason in {"content_filter", "refusal"}:
            raise LLMProviderError("provider_refused", "The LLM provider did not provide an allowed response.")

        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("invalid_provider_response", "The LLM provider returned an invalid response.")
        if message.get("refusal"):
            raise LLMProviderError("provider_refused", "The LLM provider did not provide an allowed response.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("empty_provider_response", "The LLM provider returned no content.")

        usage = _read_usage(data.get("usage"))
        return ChatCompletionResult(content=content, usage=usage)


def _read_usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LLMProviderError("invalid_provider_response", "The LLM provider returned invalid usage data.")

    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key not in value:
            continue
        token_count = value[key]
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise LLMProviderError("invalid_provider_response", "The LLM provider returned invalid usage data.")
        usage[key] = token_count
    return usage or None
