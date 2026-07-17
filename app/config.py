from __future__ import annotations

import os
from dataclasses import dataclass


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Multilingual Word Alignment API"
    app_version: str = "0.3.0"
    model: str = "xlmr"
    token_type: str = "word"
    layer: int = 8
    matching_methods: str = "mai"
    confidence_temperature: float = 0.1
    device: str = "cpu"
    eager_load: bool = False
    model_update_policy: str = "if-missing"

    def __post_init__(self) -> None:
        if self.token_type != "word":
            raise ValueError("ALIGNER_TOKEN_TYPE must be 'word' for scored token alignment")
        if self.layer < 0:
            raise ValueError("ALIGNER_LAYER must be zero or greater")
        if self.confidence_temperature <= 0.0:
            raise ValueError("ALIGNER_CONFIDENCE_TEMPERATURE must be greater than zero")
        if self.model_update_policy not in {"if-missing", "always", "offline"}:
            raise ValueError("HF_MODEL_UPDATE_POLICY must be one of: if-missing, always, offline")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.getenv("ALIGNER_MODEL", "xlmr").strip(),
            token_type=os.getenv("ALIGNER_TOKEN_TYPE", "word").strip(),
            layer=_read_int("ALIGNER_LAYER", 8),
            matching_methods=os.getenv("ALIGNER_MATCHING_METHODS", "mai").strip(),
            confidence_temperature=_read_float("ALIGNER_CONFIDENCE_TEMPERATURE", 0.1),
            device=os.getenv("ALIGNER_DEVICE", "cpu").strip(),
            eager_load=_read_bool("ALIGNER_EAGER_LOAD", False),
            model_update_policy=os.getenv("HF_MODEL_UPDATE_POLICY", "if-missing").strip().lower(),
        )
