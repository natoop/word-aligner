from __future__ import annotations

import os

import pytest

from app.llm_config import LLMProvider, LLMSettings


def _clear_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "LLM_PROVIDERS",
        "LLM_DEFAULT_PROVIDER",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_OUTPUT_TOKENS",
    ]
    for provider_id in ("DEEPSEEK", "OPENAI", "LOCAL"):
        names.extend(
            f"LLM_{provider_id}_{suffix}"
            for suffix in ("NAME", "BASE_URL", "MODEL", "API_KEY", "RESPONSE_FORMAT", "MAX_TOKENS_PARAMETER")
        )
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _write_provider_env(
    path,
    *,
    providers: str | None = "deepseek",
    extra: str = "",
    include_provider_list: bool = True,
) -> None:
    lines = []
    if include_provider_list:
        lines.append(f"LLM_PROVIDERS={providers}")
    lines.extend(
        [
            "LLM_DEEPSEEK_NAME=DeepSeek",
            "LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/",
            "LLM_DEEPSEEK_MODEL=deepseek-chat",
            "LLM_DEEPSEEK_API_KEY=file-secret-value",
            "LLM_DEEPSEEK_RESPONSE_FORMAT=json_schema",
            extra,
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_from_env_reads_dotenv_and_real_environment_overrides_without_mutating_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(
        env_file,
        extra=(
            "LLM_TIMEOUT_SECONDS=42.5\n"
            "LLM_MAX_OUTPUT_TOKENS=2048\n"
            "LLM_DEEPSEEK_MAX_TOKENS_PARAMETER=max_completion_tokens"
        ),
    )
    monkeypatch.setenv("LLM_DEEPSEEK_API_KEY", "process-secret-value")
    environment_before = os.environ.get("LLM_DEEPSEEK_API_KEY")

    settings = LLMSettings.from_env(env_file)

    assert len(settings.providers) == 1
    provider = settings.providers[0]
    assert provider.id == "deepseek"
    assert provider.name == "DeepSeek"
    assert provider.base_url == "https://api.deepseek.com/v1"
    assert provider.model == "deepseek-chat"
    assert provider.api_key == "process-secret-value"
    assert provider.response_format == "json_schema"
    assert provider.max_tokens_parameter == "max_completion_tokens"
    assert settings.default_provider == "deepseek"
    assert settings.timeout_seconds == 42.5
    assert settings.max_output_tokens == 2048
    assert os.environ.get("LLM_DEEPSEEK_API_KEY") == environment_before
    assert "process-secret-value" not in repr(provider)
    assert "file-secret-value" not in repr(settings)


def test_empty_or_missing_dotenv_starts_with_no_providers(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_environment(monkeypatch)

    settings = LLMSettings.from_env(tmp_path / "missing.env")

    assert settings.providers == ()
    assert settings.default_provider is None
    assert settings.timeout_seconds == 120
    assert settings.max_output_tokens == 8192


@pytest.mark.parametrize(
    ("providers", "include_provider_list"),
    [(None, False), ("", True)],
)
def test_configured_providers_are_discovered_without_provider_list(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    providers: str | None,
    include_provider_list: bool,
) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(
        env_file,
        providers=providers,
        include_provider_list=include_provider_list,
    )

    settings = LLMSettings.from_env(env_file)

    assert [provider.id for provider in settings.providers] == ["deepseek"]
    assert settings.default_provider == "deepseek"


def test_discovered_providers_are_sorted_and_default_can_select_one(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(
        env_file,
        include_provider_list=False,
        extra=(
            "LLM_DEFAULT_PROVIDER=openai\n"
            "LLM_OPENAI_NAME=OpenAI\n"
            "LLM_OPENAI_BASE_URL=https://api.openai.com/v1\n"
            "LLM_OPENAI_MODEL=gpt-4.1-mini"
        ),
    )

    settings = LLMSettings.from_env(env_file)

    assert [provider.id for provider in settings.providers] == ["deepseek", "openai"]
    assert settings.default_provider == "openai"


def test_dotenv_accepts_windows_utf8_bom(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(env_file)
    env_file.write_text(env_file.read_text(encoding="utf-8"), encoding="utf-8-sig")

    settings = LLMSettings.from_env(env_file)

    assert settings.default_provider == "deepseek"
    assert settings.providers[0].model == "deepseek-chat"


def test_default_provider_is_selected_when_configured(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(env_file, extra="LLM_DEFAULT_PROVIDER=deepseek")

    assert LLMSettings.from_env(env_file).default_provider == "deepseek"


def test_first_provider_is_default_when_no_default_is_configured(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(env_file)

    assert LLMSettings.from_env(env_file).default_provider == "deepseek"


@pytest.mark.parametrize(
    ("providers", "extra", "message"),
    [
        ("DeepSeek", "", "LLM_PROVIDERS"),
        ("deepseek,deepseek", "", "duplicate"),
        ("deepseek,", "", "LLM_PROVIDERS"),
        (
            "deepseek",
            "LLM_DEEPSEEK_NAME=DeepSeek\nLLM_DEEPSEEK_BASE_URL=\nLLM_DEEPSEEK_MODEL=model",
            "LLM_DEEPSEEK_BASE_URL",
        ),
        (
            "deepseek",
            "LLM_DEEPSEEK_NAME=\nLLM_DEEPSEEK_BASE_URL=https://host/v1\nLLM_DEEPSEEK_MODEL=model",
            "LLM_DEEPSEEK_NAME",
        ),
        (
            "deepseek",
            "LLM_DEEPSEEK_NAME=DeepSeek\nLLM_DEEPSEEK_BASE_URL=https://host/v1\nLLM_DEEPSEEK_MODEL=",
            "LLM_DEEPSEEK_MODEL",
        ),
        ("deepseek", "LLM_DEFAULT_PROVIDER=openai", "default provider"),
        ("deepseek", "LLM_DEEPSEEK_RESPONSE_FORMAT=xml", "RESPONSE_FORMAT"),
        ("deepseek", "LLM_DEEPSEEK_MAX_TOKENS_PARAMETER=max_completion_tokenss", "MAX_TOKENS_PARAMETER"),
        ("bad-id", "", "LLM_PROVIDERS"),
    ],
)
def test_invalid_provider_configuration_fails_safely(
    tmp_path, monkeypatch: pytest.MonkeyPatch, providers: str, extra: str, message: str
) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(env_file, providers=providers, extra=extra)

    with pytest.raises(ValueError, match=message) as error:
        LLMSettings.from_env(env_file)

    assert "file-secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?token=secret-query",
        "https://example.com/v1#secret-fragment",
        "https://example.com:invalid/v1",
    ],
)
def test_base_url_rejects_unsafe_or_invalid_components(base_url: str) -> None:
    with pytest.raises(ValueError) as error:
        LLMProvider(id="local", name="Local", base_url=base_url, model="test")

    assert "secret-query" not in str(error.value)
    assert "secret-fragment" not in str(error.value)
    assert "password" not in str(error.value)


@pytest.mark.parametrize(
    ("timeout", "tokens"),
    [("0", "8192"), ("inf", "8192"), ("601", "8192"), ("120", "0"), ("120", "65537"), ("120", "nope")],
)
def test_timeout_and_token_limits_must_be_positive_finite_and_bounded(
    tmp_path, monkeypatch: pytest.MonkeyPatch, timeout: str, tokens: str
) -> None:
    _clear_llm_environment(monkeypatch)
    env_file = tmp_path / ".env"
    _write_provider_env(env_file, extra=f"LLM_TIMEOUT_SECONDS={timeout}\nLLM_MAX_OUTPUT_TOKENS={tokens}")

    with pytest.raises(ValueError):
        LLMSettings.from_env(env_file)


def test_provider_id_is_validated_and_key_is_hidden_from_repr() -> None:
    with pytest.raises(ValueError, match="Provider id"):
        LLMProvider(id="Local", name="Local", base_url="http://localhost/v1", model="test")

    provider = LLMProvider(
        id="local",
        name="Local",
        base_url="http://localhost:8000/v1/",
        model="test",
        api_key="top-secret",
    )
    assert provider.base_url == "http://localhost:8000/v1"
    assert "top-secret" not in repr(provider)


def test_provider_can_select_completion_token_parameter() -> None:
    provider = LLMProvider(
        id="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        max_tokens_parameter="max_completion_tokens",
    )

    assert provider.max_tokens_parameter == "max_completion_tokens"
