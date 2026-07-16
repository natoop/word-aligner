from __future__ import annotations

import logging
import threading
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from importlib.util import find_spec
from typing import Protocol

from app.config import Settings
from app.model_cache import configure_huggingface_access
from app.schemas import (
    AlignmentGroup,
    AlignmentLink,
    AlignmentLinkOrigin,
    AlignmentRequest,
    AlignmentResponse,
    SentenceAlignment,
    Token,
)
from app.tokenization import WordTokenizer

RawAlignment = Mapping[str, Iterable[tuple[int, int]]]
logger = logging.getLogger(__name__)


class WordAlignerBackend(Protocol):
    def get_word_aligns(self, source_tokens: list[str], target_tokens: list[str]) -> RawAlignment: ...


class AlignmentModelError(RuntimeError):
    """The neural alignment model could not be loaded."""


class AlignmentProcessingError(RuntimeError):
    """A sentence pair could not be aligned."""


class SimAlignBackend:
    def __init__(self, settings: Settings) -> None:
        model_access = configure_huggingface_access(settings.model, settings.model_update_policy)
        if model_access.offline:
            logger.info(
                "Using cached Hugging Face model in offline mode: model=%s snapshot=%s",
                model_access.model_id,
                model_access.snapshot_path,
            )
        else:
            logger.info(
                "Hugging Face model cache is incomplete; downloads are allowed: model=%s",
                model_access.model_id,
            )

        try:
            import torch
        except ImportError as exc:
            raise AlignmentModelError("The 'torch' package is not installed") from exc

        _ensure_device_available(settings.device, torch.cuda.is_available())

        if find_spec("sentencepiece") is None:
            raise AlignmentModelError(
                "The 'sentencepiece' package is required by the XLM-R tokenizer. "
                "Install production dependencies from requirements.txt."
            )

        try:
            from simalign import SentenceAligner
        except ImportError as exc:
            raise AlignmentModelError(f"Unable to import SimAlign: {exc}") from exc

        self._aligner = SentenceAligner(
            model=settings.model,
            token_type=settings.token_type,
            matching_methods=settings.matching_methods,
            device=settings.device,
        )

    def get_word_aligns(self, source_tokens: list[str], target_tokens: list[str]) -> RawAlignment:
        return self._aligner.get_word_aligns(source_tokens, target_tokens)


def _ensure_device_available(device: str, cuda_available: bool) -> None:
    if device.lower().startswith("cuda") and not cuda_available:
        raise AlignmentModelError(
            f"ALIGNER_DEVICE is set to '{device}', but CUDA is unavailable. "
            "Set ALIGNER_DEVICE=cpu or configure an NVIDIA driver and container GPU access."
        )


BackendFactory = Callable[[], WordAlignerBackend]


class AlignmentService:
    def __init__(
        self,
        settings: Settings,
        tokenizer: WordTokenizer | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self._settings = settings
        self._tokenizer = tokenizer or WordTokenizer()
        self._backend_factory = backend_factory or (lambda: SimAlignBackend(settings))
        self._backend: WordAlignerBackend | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._backend is not None

    def warm_up(self) -> None:
        self._get_backend()

    def align(self, request: AlignmentRequest) -> AlignmentResponse:
        backend = self._get_backend()
        sentence_alignments: list[SentenceAlignment] = []

        for pair_index, pair in enumerate(request.sentence_pairs):
            source_tokens = self._tokenizer.tokenize(pair.source, request.source_language)
            target_tokens = self._tokenizer.tokenize(pair.target, request.target_language)

            try:
                with self._inference_lock:
                    raw_alignments = backend.get_word_aligns(
                        [token.text for token in source_tokens],
                        [token.text for token in target_tokens],
                    )
            except Exception as exc:
                raise AlignmentProcessingError(f"Failed to align sentence pair at index {pair_index}") from exc

            method_links = raw_alignments.get(request.method)
            if method_links is None:
                raise AlignmentProcessingError(
                    f"Alignment method '{request.method}' is not enabled by ALIGNER_MATCHING_METHODS"
                )

            links = _normalize_links(method_links, len(source_tokens), len(target_tokens))
            links = _apply_protected_token_alignment(source_tokens, target_tokens, links)
            rule_links = {
                (source_index, target_index)
                for source_index, target_index in links
                if source_tokens[source_index].is_protected or target_tokens[target_index].is_protected
            }

            repaired_links: set[tuple[int, int]] = set()
            if request.repair is not None and request.repair.enabled:
                fallback_method_links = raw_alignments.get("mwmf")
                if fallback_method_links is None:
                    raise AlignmentProcessingError(
                        "Conservative repair requires 'mwmf'; include 'm' in ALIGNER_MATCHING_METHODS"
                    )
                fallback_links = _normalize_links(
                    fallback_method_links,
                    len(source_tokens),
                    len(target_tokens),
                )
                repaired_links = _repair_unaligned_links(
                    source_tokens=source_tokens,
                    target_tokens=target_tokens,
                    base_links=links,
                    fallback_links=fallback_links,
                    max_position_distance=request.repair.max_position_distance,
                )
                links.update(repaired_links)

            link_origins: dict[tuple[int, int], AlignmentLinkOrigin] = {
                link: "model" for link in links
            }
            link_origins.update({link: "rule" for link in rule_links})
            link_origins.update({link: "repaired" for link in repaired_links})
            sentence_alignments.append(
                _build_sentence_alignment(
                    pair_index=pair_index,
                    pair_id=pair.id,
                    source=pair.source,
                    target=pair.target,
                    source_tokens=source_tokens,
                    target_tokens=target_tokens,
                    links=links,
                    link_origins=link_origins,
                )
            )

        return AlignmentResponse(
            source_language=request.source_language,
            target_language=request.target_language,
            model=self._settings.model,
            method=request.method,
            sentence_alignments=sentence_alignments,
        )

    def _get_backend(self) -> WordAlignerBackend:
        if self._backend is not None:
            return self._backend

        with self._load_lock:
            if self._backend is not None:
                return self._backend
            try:
                self._backend = self._backend_factory()
            except AlignmentModelError:
                raise
            except Exception as exc:
                raise AlignmentModelError("Unable to initialize the SimAlign model") from exc
            return self._backend


def _normalize_links(
    raw_links: Iterable[tuple[int, int]], source_count: int, target_count: int
) -> set[tuple[int, int]]:
    links: set[tuple[int, int]] = set()
    for raw_source_index, raw_target_index in raw_links:
        source_index = int(raw_source_index)
        target_index = int(raw_target_index)
        if 0 <= source_index < source_count and 0 <= target_index < target_count:
            links.add((source_index, target_index))
    return links


def _apply_protected_token_alignment(
    source_tokens: list[Token], target_tokens: list[Token], links: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    source_protected = {token.index for token in source_tokens if token.is_protected}
    target_protected = {token.index for token in target_tokens if token.is_protected}

    if not source_protected and not target_protected:
        return links

    result = {
        (source_index, target_index)
        for source_index, target_index in links
        if source_index not in source_protected and target_index not in target_protected
    }

    source_by_text: dict[str, list[int]] = defaultdict(list)
    target_by_text: dict[str, list[int]] = defaultdict(list)
    for token in source_tokens:
        if token.is_protected:
            source_by_text[token.text].append(token.index)
    for token in target_tokens:
        if token.is_protected:
            target_by_text[token.text].append(token.index)

    for token_text, source_indices in source_by_text.items():
        for source_index, target_index in zip(source_indices, target_by_text.get(token_text, []), strict=False):
            result.add((source_index, target_index))

    return result


def _repair_unaligned_links(
    *,
    source_tokens: list[Token],
    target_tokens: list[Token],
    base_links: set[tuple[int, int]],
    fallback_links: set[tuple[int, int]],
    max_position_distance: float,
) -> set[tuple[int, int]]:
    aligned_source = {source_index for source_index, _ in base_links}
    aligned_target = {target_index for _, target_index in base_links}
    candidates: list[tuple[float, int, int]] = []

    for source_index, target_index in fallback_links:
        if source_index in aligned_source or target_index in aligned_target:
            continue
        if not _is_repairable_token(source_tokens[source_index]):
            continue
        if not _is_repairable_token(target_tokens[target_index]):
            continue

        position_distance = abs(
            _relative_position(source_index, len(source_tokens))
            - _relative_position(target_index, len(target_tokens))
        )
        if position_distance <= max_position_distance:
            candidates.append((position_distance, source_index, target_index))

    repaired_links: set[tuple[int, int]] = set()
    repaired_source: set[int] = set()
    repaired_target: set[int] = set()
    for _, source_index, target_index in sorted(candidates):
        if source_index in repaired_source or target_index in repaired_target:
            continue
        repaired_links.add((source_index, target_index))
        repaired_source.add(source_index)
        repaired_target.add(target_index)

    return repaired_links


def _relative_position(index: int, token_count: int) -> float:
    return (index + 0.5) / token_count


def _is_repairable_token(token: Token) -> bool:
    if token.is_protected:
        return False
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in token.text)


def _build_sentence_alignment(
    *,
    pair_index: int,
    pair_id: str | None,
    source: str,
    target: str,
    source_tokens: list[Token],
    target_tokens: list[Token],
    links: set[tuple[int, int]],
    link_origins: Mapping[tuple[int, int], AlignmentLinkOrigin],
) -> SentenceAlignment:
    sorted_links = sorted(links)
    groups = _build_alignment_groups(source_tokens, target_tokens, sorted_links, link_origins)
    aligned_source = {source_index for source_index, _ in sorted_links}
    aligned_target = {target_index for _, target_index in sorted_links}

    return SentenceAlignment(
        index=pair_index,
        id=pair_id,
        source=source,
        target=target,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        links=[
            AlignmentLink(
                source_index=source_index,
                target_index=target_index,
                origin=link_origins[(source_index, target_index)],
            )
            for source_index, target_index in sorted_links
        ],
        alignment_groups=groups,
        unaligned_source_indices=[token.index for token in source_tokens if token.index not in aligned_source],
        unaligned_target_indices=[token.index for token in target_tokens if token.index not in aligned_target],
    )


def _build_alignment_groups(
    source_tokens: list[Token],
    target_tokens: list[Token],
    links: list[tuple[int, int]],
    link_origins: Mapping[tuple[int, int], AlignmentLinkOrigin],
) -> list[AlignmentGroup]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for source_index, target_index in links:
        source_node = ("source", source_index)
        target_node = ("target", target_index)
        adjacency[source_node].add(target_node)
        adjacency[target_node].add(source_node)

    visited: set[tuple[str, int]] = set()
    components: list[tuple[list[int], list[int]]] = []

    for node in sorted(adjacency, key=lambda item: (item[1], item[0])):
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        source_indices: set[int] = set()
        target_indices: set[int] = set()

        while queue:
            side, index = queue.popleft()
            if side == "source":
                source_indices.add(index)
            else:
                target_indices.add(index)

            for neighbor in adjacency[(side, index)]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append((sorted(source_indices), sorted(target_indices)))

    components.sort(
        key=lambda component: (
            component[0][0] if component[0] else len(source_tokens),
            component[1][0] if component[1] else len(target_tokens),
        )
    )

    groups: list[AlignmentGroup] = []
    for source_indices, target_indices in components:
        component_links = [
            AlignmentLink(
                source_index=source_index,
                target_index=target_index,
                origin=link_origins[(source_index, target_index)],
            )
            for source_index, target_index in links
            if source_index in source_indices and target_index in target_indices
        ]
        groups.append(
            AlignmentGroup(
                type=_classify_group(len(source_indices), len(target_indices)),
                source_indices=source_indices,
                target_indices=target_indices,
                source_tokens=[source_tokens[index].text for index in source_indices],
                target_tokens=[target_tokens[index].text for index in target_indices],
                links=component_links,
            )
        )
    return groups


def _classify_group(source_count: int, target_count: int) -> str:
    if source_count == 1 and target_count == 1:
        return "one-to-one"
    if source_count == 1:
        return "one-to-many"
    if target_count == 1:
        return "many-to-one"
    return "many-to-many"
