from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


_MODEL_ALIASES = {
    "bert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}
_DIRECT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_TOKENIZER_FILES = (
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.json",
    "vocab.txt",
    "vocab.json",
)
_OFFLINE_ENVIRONMENT_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUE_VALUES = {"1", "on", "true", "yes"}


@dataclass(frozen=True, slots=True)
class HuggingFaceAccess:
    model_id: str
    offline: bool
    cache_complete: bool
    snapshot_path: Path | None


def configure_huggingface_access(model: str, update_policy: str) -> HuggingFaceAccess:
    model_id = _MODEL_ALIASES.get(model, model)
    snapshot_path = _find_complete_model_snapshot(model_id)
    cache_complete = snapshot_path is not None
    explicitly_offline = any(_is_truthy(os.getenv(name)) for name in _OFFLINE_ENVIRONMENT_VARIABLES)

    if update_policy == "always":
        offline = False
    elif update_policy == "offline":
        offline = True
    else:
        offline = cache_complete or explicitly_offline

    if offline:
        for name in _OFFLINE_ENVIRONMENT_VARIABLES:
            os.environ[name] = "1"
    else:
        for name in _OFFLINE_ENVIRONMENT_VARIABLES:
            os.environ.pop(name, None)

    return HuggingFaceAccess(
        model_id=model_id,
        offline=offline,
        cache_complete=cache_complete,
        snapshot_path=snapshot_path,
    )


def _find_complete_model_snapshot(model_id: str) -> Path | None:
    local_model_path = Path(model_id)
    if local_model_path.is_dir() and _is_complete_snapshot(local_model_path):
        return local_model_path.resolve()

    repository_cache = _hub_cache_root() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repository_cache.is_dir():
        return None

    for snapshot_path in sorted(repository_cache.iterdir(), reverse=True):
        if snapshot_path.is_dir() and _is_complete_snapshot(snapshot_path):
            return snapshot_path
    return None


def _hub_cache_root() -> Path:
    explicit_hub_cache = os.getenv("HF_HUB_CACHE") or os.getenv("TRANSFORMERS_CACHE")
    if explicit_hub_cache:
        return Path(explicit_hub_cache).expanduser()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_complete_snapshot(snapshot_path: Path) -> bool:
    return (
        _is_nonempty_file(snapshot_path / "config.json")
        and _has_complete_weights(snapshot_path)
        and any(_is_nonempty_file(snapshot_path / name) for name in _TOKENIZER_FILES)
    )


def _has_complete_weights(snapshot_path: Path) -> bool:
    if any(_is_nonempty_file(snapshot_path / name) for name in _DIRECT_WEIGHT_FILES):
        return True

    for index_name in _WEIGHT_INDEX_FILES:
        index_path = snapshot_path / index_name
        if not _is_nonempty_file(index_path):
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(index["weight_map"].values())
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if shard_names and all(_is_nonempty_file(snapshot_path / name) for name in shard_names):
            return True
    return False


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE_VALUES
