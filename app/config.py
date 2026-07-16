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


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "Multilingual Word Alignment API"
    app_version: str = "0.1.0"
    model: str = "xlmr"
    token_type: str = "bpe"
    matching_methods: str = "mai"
    device: str = "cpu"
    eager_load: bool = False
    model_update_policy: str = "if-missing"

    def __post_init__(self) -> None:
        if self.model_update_policy not in {"if-missing", "always", "offline"}:
            raise ValueError("HF_MODEL_UPDATE_POLICY must be one of: if-missing, always, offline")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model=os.getenv("ALIGNER_MODEL", "xlmr").strip(),
            token_type=os.getenv("ALIGNER_TOKEN_TYPE", "bpe").strip(),
            matching_methods=os.getenv("ALIGNER_MATCHING_METHODS", "mai").strip(),
            device=os.getenv("ALIGNER_DEVICE", "cpu").strip(),
            eager_load=_read_bool("ALIGNER_EAGER_LOAD", False),
            model_update_policy=os.getenv("HF_MODEL_UPDATE_POLICY", "if-missing").strip().lower(),
        )
