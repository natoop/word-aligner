from __future__ import annotations

import pytest

from app.config import Settings


def test_settings_read_token_embedding_scoring_options_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALIGNER_TOKEN_TYPE", "word")
    monkeypatch.setenv("ALIGNER_LAYER", "10")
    monkeypatch.setenv("ALIGNER_CONFIDENCE_TEMPERATURE", "0.2")

    settings = Settings.from_env()

    assert settings.token_type == "word"
    assert settings.layer == 10
    assert settings.confidence_temperature == 0.2


def test_settings_require_word_level_embedding_aggregation() -> None:
    with pytest.raises(ValueError, match="ALIGNER_TOKEN_TYPE"):
        Settings(token_type="bpe")


def test_settings_reject_an_invalid_embedding_layer() -> None:
    with pytest.raises(ValueError, match="ALIGNER_LAYER"):
        Settings(layer=-1)


def test_settings_reject_a_nonpositive_confidence_temperature() -> None:
    with pytest.raises(ValueError, match="ALIGNER_CONFIDENCE_TEMPERATURE"):
        Settings(confidence_temperature=0.0)
