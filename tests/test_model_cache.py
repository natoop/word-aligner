from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import Settings
from app.model_cache import configure_huggingface_access


def _create_complete_snapshot(hf_home: Path) -> Path:
    snapshot = hf_home / "hub" / "models--xlm-roberta-base" / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"model-weights")
    (snapshot / "sentencepiece.bpe.model").write_bytes(b"tokenizer")
    return snapshot


def _clear_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


def test_if_missing_policy_enables_offline_mode_for_a_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _create_complete_snapshot(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _clear_offline_environment(monkeypatch)

    access = configure_huggingface_access("xlmr", "if-missing")

    assert access.cache_complete is True
    assert access.offline is True
    assert access.snapshot_path == snapshot
    assert access.model_id == "xlm-roberta-base"
    assert access.snapshot_path is not None
    assert access.snapshot_path.name == "revision-1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_if_missing_policy_allows_download_for_an_incomplete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "hub" / "models--xlm-roberta-base" / "snapshots" / "partial"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "sentencepiece.bpe.model").write_bytes(b"tokenizer")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _clear_offline_environment(monkeypatch)

    access = configure_huggingface_access("xlmr", "if-missing")

    assert access.cache_complete is False
    assert access.offline is False
    assert access.snapshot_path is None
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_always_policy_keeps_update_checks_enabled_for_a_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_complete_snapshot(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    access = configure_huggingface_access("xlmr", "always")

    assert access.cache_complete is True
    assert access.offline is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_offline_policy_does_not_require_a_preexisting_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _clear_offline_environment(monkeypatch)

    access = configure_huggingface_access("xlmr", "offline")

    assert access.cache_complete is False
    assert access.offline is True


def test_settings_reject_an_unknown_model_update_policy() -> None:
    with pytest.raises(ValueError, match="HF_MODEL_UPDATE_POLICY"):
        Settings(model_update_policy="sometimes")
