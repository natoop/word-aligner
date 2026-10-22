from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

_PROVIDER_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_RESPONSE_FORMATS = {"json_object", "json_schema", "none"}
_MAX_TOKENS_PARAMETERS = {"max_tokens", "max_completion_tokens"}
_MAX_TIMEOUT_SECONDS = 600.0
_MAX_OUTPUT_TOKENS = 65_536


def _validate_base_url(value: str, variable_name: str) -> str:
    if not value or "?" in value or "#" in value or any(character.isspace() for character in value):
        raise ValueError(f"{variable_name} must be a valid HTTP or HTTPS base URL")

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError(f"{variable_name} must be a valid HTTP or HTTPS base URL") from None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or (port is not None and not 0 <= port <= 65_535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{variable_name} must be an HTTP or HTTPS base URL without credentials, query, or fragment")

    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class LLMProvider:
    id: str
    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    response_format: str = "json_object"
    max_tokens_parameter: str = "max_tokens"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _PROVIDER_ID.fullmatch(self.id):
            raise ValueError("Provider id must match [a-z][a-z0-9_]*")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Provider name is required")
        if not isinstance(self.base_url, str):
            raise ValueError("Provider base URL must be a valid HTTP or HTTPS base URL")
        object.__setattr__(self, "base_url", _validate_base_url(self.base_url.strip(), "Provider base URL"))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Provider model is required")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "model", self.model.strip())
        if not isinstance(self.api_key, str):
            raise ValueError("Provider API key must be a string")
        object.__setattr__(self, "api_key", self.api_key.strip())
        if not isinstance(self.response_format, str) or self.response_format not in _RESPONSE_FORMATS:
            raise ValueError("Provider response format must be json_object, json_schema, or none")
        if not isinstance(self.max_tokens_parameter, str) or self.max_tokens_parameter not in _MAX_TOKENS_PARAMETERS:
            raise ValueError("Provider max tokens parameter must be max_tokens or max_completion_tokens")


@dataclass(frozen=True, slots=True)
class LLMSettings:
    providers: tuple[LLMProvider, ...] = ()
    default_provider: str | None = None
    timeout_seconds: float = 120.0
    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.providers, tuple) or any(
            not isinstance(provider, LLMProvider) for provider in self.providers
        ):
            raise ValueError("LLM providers must be a tuple of provider configurations")

        provider_ids = [provider.id for provider in self.providers]
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("LLM provider ids must be unique")
        if self.default_provider is None and provider_ids:
            object.__setattr__(self, "default_provider", provider_ids[0])
        if self.default_provider is not None:
            if not isinstance(self.default_provider, str) or not _PROVIDER_ID.fullmatch(self.default_provider):
                raise ValueError("LLM default provider id must match [a-z][a-z0-9_]*")
            if self.default_provider not in provider_ids:
                raise ValueError("LLM default provider must name a configured provider")

        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number no greater than 600")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= _MAX_TIMEOUT_SECONDS:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number no greater than 600")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or not 0 < self.max_output_tokens <= _MAX_OUTPUT_TOKENS
        ):
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be an integer from 1 to 65536")

    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> LLMSettings:
        selected_env_file = Path(env_file) if env_file is not None else Path(__file__).resolve().parent.parent / ".env"
        try:
            file_values = dotenv_values(selected_env_file, interpolate=False, encoding="utf-8-sig")
        except (OSError, UnicodeError, ValueError):
            raise ValueError("Unable to read LLM environment file") from None

        # Real process environment variables always take precedence over the project .env file.
        values: dict[str, str | None] = {**file_values, **os.environ}
        raw_provider_ids = values.get("LLM_PROVIDERS")
        provider_ids = _parse_provider_ids(raw_provider_ids) or _discover_provider_ids(values)
        providers: list[LLMProvider] = []

        for provider_id in provider_ids:
            prefix = f"LLM_{provider_id.upper()}_"
            name = _required_value(values, f"{prefix}NAME")
            base_url_value = _required_value(values, f"{prefix}BASE_URL")
            model = _required_value(values, f"{prefix}MODEL")
            api_key = (values.get(f"{prefix}API_KEY") or "").strip()
            response_format = (values.get(f"{prefix}RESPONSE_FORMAT") or "json_object").strip()
            if response_format not in _RESPONSE_FORMATS:
                raise ValueError(f"{prefix}RESPONSE_FORMAT must be json_object, json_schema, or none")
            max_tokens_parameter = (values.get(f"{prefix}MAX_TOKENS_PARAMETER") or "max_tokens").strip()
            if max_tokens_parameter not in _MAX_TOKENS_PARAMETERS:
                raise ValueError(f"{prefix}MAX_TOKENS_PARAMETER must be max_tokens or max_completion_tokens")

            providers.append(
                LLMProvider(
                    id=provider_id,
                    name=name,
                    base_url=_validate_base_url(base_url_value.strip(), f"{prefix}BASE_URL"),
                    model=model,
                    api_key=api_key,
                    response_format=response_format,
                    max_tokens_parameter=max_tokens_parameter,
                )
            )

        default_provider = (values.get("LLM_DEFAULT_PROVIDER") or "").strip() or None
        timeout_seconds = _read_timeout(values.get("LLM_TIMEOUT_SECONDS", "120"))
        max_output_tokens = _read_max_output_tokens(values.get("LLM_MAX_OUTPUT_TOKENS", "8192"))
        return cls(
            providers=tuple(providers),
            default_provider=default_provider,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )


_PROVIDER_DISCOVERY_VARIABLE = re.compile(r"^LLM_([A-Z][A-Z0-9_]*)_(?:NAME|BASE_URL|MODEL)$")


def _discover_provider_ids(values: dict[str, str | None]) -> tuple[str, ...]:
    provider_ids = {
        match.group(1).lower()
        for variable_name, value in values.items()
        if (match := _PROVIDER_DISCOVERY_VARIABLE.fullmatch(variable_name))
        and value is not None
        and value.strip()
    }
    return tuple(sorted(provider_ids))


def _parse_provider_ids(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()

    parts = value.split(",")
    provider_ids = tuple(part.strip() for part in parts)
    if any(not provider_id or not _PROVIDER_ID.fullmatch(provider_id) for provider_id in provider_ids):
        raise ValueError("LLM_PROVIDERS must be a comma-separated list of ids matching [a-z][a-z0-9_]*")
    if len(set(provider_ids)) != len(provider_ids):
        raise ValueError("LLM_PROVIDERS must not contain duplicate ids")
    return provider_ids


def _required_value(values: dict[str, str | None], variable_name: str) -> str:
    value = values.get(variable_name)
    if value is None or not value.strip():
        raise ValueError(f"{variable_name} is required for a configured LLM provider")
    return value.strip()


def _read_timeout(value: str | None) -> float:
    try:
        timeout = float(value) if value is not None else 120.0
    except (TypeError, ValueError):
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number no greater than 600") from None
    if not math.isfinite(timeout) or not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number no greater than 600")
    return timeout


def _read_max_output_tokens(value: str | None) -> int:
    try:
        max_output_tokens = int(value) if value is not None else 8192
    except (TypeError, ValueError):
        raise ValueError("LLM_MAX_OUTPUT_TOKENS must be an integer from 1 to 65536") from None
    if not 0 < max_output_tokens <= _MAX_OUTPUT_TOKENS:
        raise ValueError("LLM_MAX_OUTPUT_TOKENS must be an integer from 1 to 65536")
    return max_output_tokens
