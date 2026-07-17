from __future__ import annotations

import logging
import math
import threading
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Protocol

import numpy as np

from app.config import Settings
from app.model_cache import configure_huggingface_access
from app.schemas import (
    AlignmentGroup,
    AlignmentGroupOrigin,
    AlignmentLink,
    AlignmentLinkOrigin,
    AlignmentRequest,
    AlignmentResponse,
    SentenceAlignment,
    Token,
)
from app.span_refinement import RefinedSpan, refine_alignment_spans
from app.tokenization import WordTokenizer

RawLink = tuple[int, int]
RawAlignment = Mapping[str, Iterable[RawLink]]
ScoreMatrix = tuple[tuple[float, ...], ...]
CONFIDENCE_METHOD = "bidirectional-margin-span-v2"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenEmbeddingAlignment:
    """Word-level model output from one contextual embedding inference."""

    alignments: RawAlignment
    similarities: ScoreMatrix
    source_to_target_probabilities: ScoreMatrix
    target_to_source_probabilities: ScoreMatrix
    source_embeddings: np.ndarray | None = None
    target_embeddings: np.ndarray | None = None
    relative_similarities: ScoreMatrix | None = None


@dataclass(frozen=True, slots=True)
class LinkScore:
    similarity: float
    confidence: float


class WordAlignerBackend(Protocol):
    def align_tokens(self, source_tokens: list[str], target_tokens: list[str]) -> TokenEmbeddingAlignment: ...


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
            layer=settings.layer,
        )
        self._confidence_temperature = settings.confidence_temperature

    def align_tokens(self, source_tokens: list[str], target_tokens: list[str]) -> TokenEmbeddingAlignment:
        if not source_tokens or not target_tokens:
            empty_matrix: ScoreMatrix = tuple(tuple() for _ in source_tokens)
            return TokenEmbeddingAlignment(
                alignments={method: () for method in self._aligner.matching_methods},
                similarities=empty_matrix,
                source_to_target_probabilities=empty_matrix,
                target_to_source_probabilities=empty_matrix,
                relative_similarities=empty_matrix,
            )

        tokenizer = self._aligner.embed_loader.tokenizer
        source_subwords = [_tokenize_model_word(tokenizer, token) for token in source_tokens]
        target_subwords = [_tokenize_model_word(tokenizer, token) for token in target_tokens]
        subword_lists = [
            [subword for word_subwords in side for subword in word_subwords]
            for side in (source_subwords, target_subwords)
        ]

        vectors = self._aligner.embed_loader.get_embed_list([source_tokens, target_tokens]).cpu().detach().numpy()
        if any(vectors.shape[1] < len(subwords) for subwords in subword_lists):
            raise ValueError("The tokenized sentence exceeds the embedding model's maximum sequence length")

        subword_vectors = [vectors[side, : len(subword_lists[side])] for side in (0, 1)]
        source_word_vectors, target_word_vectors = self._aligner.average_embeds_over_words(
            subword_vectors,
            [source_subwords, target_subwords],
        )
        similarity_matrix = np.clip(
            self._aligner.get_similarity(source_word_vectors, target_word_vectors),
            0.0,
            1.0,
        )
        relative_similarity_matrix = _relative_similarity_matrix(
            similarity_matrix,
            temperature=self._confidence_temperature,
        )

        forward, reverse = self._aligner.get_alignment_matrix(similarity_matrix)
        alignment_matrices: dict[str, np.ndarray] = {
            "fwd": forward,
            "rev": reverse,
            "inter": forward * reverse,
        }
        if "mwmf" in self._aligner.matching_methods:
            alignment_matrices["mwmf"] = self._aligner.get_max_weight_match(similarity_matrix)
        if "itermax" in self._aligner.matching_methods:
            alignment_matrices["itermax"] = self._aligner.iter_max(similarity_matrix)

        alignments = {
            method: tuple(
                (int(source_index), int(target_index))
                for source_index, target_index in np.argwhere(alignment_matrices[method] > 0)
            )
            for method in self._aligner.matching_methods
        }
        source_probabilities = _softmax_matrix(
            relative_similarity_matrix,
            axis=1,
            temperature=self._confidence_temperature,
        )
        target_probabilities = _softmax_matrix(
            relative_similarity_matrix,
            axis=0,
            temperature=self._confidence_temperature,
        )
        return TokenEmbeddingAlignment(
            alignments=alignments,
            similarities=_freeze_matrix(similarity_matrix),
            source_to_target_probabilities=_freeze_matrix(source_probabilities),
            target_to_source_probabilities=_freeze_matrix(target_probabilities),
            source_embeddings=_freeze_embeddings(source_word_vectors),
            target_embeddings=_freeze_embeddings(target_word_vectors),
            relative_similarities=_freeze_matrix(relative_similarity_matrix),
        )


def _tokenize_model_word(tokenizer: object, token: str) -> list[str]:
    subwords = list(tokenizer.tokenize(token))  # type: ignore[attr-defined]
    if subwords:
        return subwords

    unknown_token = getattr(tokenizer, "unk_token", None)
    if unknown_token:
        return [str(unknown_token)]
    raise ValueError(f"The embedding tokenizer produced no subwords for token: {token!r}")


def _softmax_matrix(matrix: np.ndarray, *, axis: int, temperature: float) -> np.ndarray:
    scaled = matrix / temperature
    scaled = scaled - np.max(scaled, axis=axis, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _relative_similarity_matrix(matrix: np.ndarray, *, temperature: float) -> np.ndarray:
    """Convert absolute cosine scores into CSLS-style local evidence."""

    if matrix.size == 0:
        return matrix.copy()
    row_k = min(3, matrix.shape[1])
    column_k = min(3, matrix.shape[0])
    row_neighborhood = np.partition(matrix, matrix.shape[1] - row_k, axis=1)[:, -row_k:]
    column_neighborhood = np.partition(matrix, matrix.shape[0] - column_k, axis=0)[-column_k:, :]
    row_baseline = row_neighborhood.mean(axis=1, keepdims=True)
    column_baseline = column_neighborhood.mean(axis=0, keepdims=True)
    relative = 2.0 * matrix - row_baseline - column_baseline
    scaled = np.clip(relative / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _freeze_matrix(matrix: np.ndarray) -> ScoreMatrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _freeze_embeddings(vectors: object) -> np.ndarray:
    embeddings = np.asarray(vectors, dtype=float).copy()
    embeddings.setflags(write=False)
    return embeddings


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
                    embedding_result = backend.align_tokens(
                        [token.text for token in source_tokens],
                        [token.text for token in target_tokens],
                    )
                _validate_embedding_result(
                    embedding_result,
                    source_count=len(source_tokens),
                    target_count=len(target_tokens),
                )
            except Exception as exc:
                raise AlignmentProcessingError(f"Failed to align sentence pair at index {pair_index}") from exc

            normalized_alignments = {
                method: _normalize_links(raw_links, len(source_tokens), len(target_tokens))
                for method, raw_links in embedding_result.alignments.items()
            }
            method_links = normalized_alignments.get(request.method)
            if method_links is None:
                raise AlignmentProcessingError(
                    f"Alignment method '{request.method}' is not enabled by ALIGNER_MATCHING_METHODS"
                )

            links = set(method_links)
            links = _apply_protected_token_alignment(source_tokens, target_tokens, links)
            rule_links = {
                (source_index, target_index)
                for source_index, target_index in links
                if source_tokens[source_index].is_protected or target_tokens[target_index].is_protected
            }
            link_scores = {
                link: _score_model_link(embedding_result, link, normalized_alignments)
                for link in links
                if link not in rule_links
            }
            link_scores.update({link: LinkScore(similarity=1.0, confidence=1.0) for link in rule_links})

            repaired_link_scores: dict[tuple[int, int], LinkScore] = {}
            if request.repair is not None and request.repair.enabled:
                fallback_method_links = normalized_alignments.get("mwmf")
                if fallback_method_links is None:
                    raise AlignmentProcessingError(
                        "Conservative repair requires 'mwmf'; include 'm' in ALIGNER_MATCHING_METHODS"
                    )
                repaired_link_scores = _repair_unaligned_links(
                    source_tokens=source_tokens,
                    target_tokens=target_tokens,
                    base_links=links,
                    fallback_links=fallback_method_links,
                    embedding_result=embedding_result,
                    normalized_alignments=normalized_alignments,
                    max_position_distance=request.repair.max_position_distance,
                    min_similarity=request.repair.min_similarity,
                    min_confidence=request.repair.min_confidence,
                )
                links.update(repaired_link_scores)
                link_scores.update(repaired_link_scores)

            link_origins: dict[tuple[int, int], AlignmentLinkOrigin] = {link: "model" for link in links}
            link_origins.update({link: "rule" for link in rule_links})
            link_origins.update({link: "repaired" for link in repaired_link_scores})

            refined_spans: Sequence[RefinedSpan] = ()
            grouped_links: set[tuple[int, int]] = set()
            if request.repair is not None and request.repair.enabled:
                span_refinement = refine_alignment_spans(
                    source_tokens=source_tokens,
                    target_tokens=target_tokens,
                    links=links,
                    similarities=embedding_result.similarities,
                    relative_similarities=(embedding_result.relative_similarities or embedding_result.similarities),
                    link_confidences={link: score.confidence for link, score in link_scores.items()},
                    strategy=request.repair.strategy,
                    max_source_span=request.repair.max_source_span,
                    max_target_span=request.repair.max_target_span,
                    min_score_gain=request.repair.min_score_gain,
                    min_span_coverage=request.repair.min_span_coverage,
                    source_embeddings=embedding_result.source_embeddings,
                    target_embeddings=embedding_result.target_embeddings,
                )
                links = set(span_refinement.links)
                refined_spans = span_refinement.groups
                grouped_links = set(span_refinement.grouped_links)

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
                    link_scores=link_scores,
                    refined_spans=refined_spans,
                    grouped_links=grouped_links,
                )
            )

        return AlignmentResponse(
            source_language=request.source_language,
            target_language=request.target_language,
            model=self._settings.model,
            embedding_layer=self._settings.layer,
            method=request.method,
            confidence_method=CONFIDENCE_METHOD,
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


def _validate_embedding_result(
    result: TokenEmbeddingAlignment,
    *,
    source_count: int,
    target_count: int,
) -> None:
    matrices = {
        "similarities": result.similarities,
        "source_to_target_probabilities": result.source_to_target_probabilities,
        "target_to_source_probabilities": result.target_to_source_probabilities,
    }
    if result.relative_similarities is not None:
        matrices["relative_similarities"] = result.relative_similarities
    for name, matrix in matrices.items():
        if len(matrix) != source_count or any(len(row) != target_count for row in matrix):
            raise ValueError(f"Backend {name} matrix has an invalid shape; expected {source_count}x{target_count}")
        for row in matrix:
            for value in row:
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Backend {name} matrix contains a score outside 0..1")

    embeddings = {
        "source_embeddings": (result.source_embeddings, source_count),
        "target_embeddings": (result.target_embeddings, target_count),
    }
    embedding_dimensions: set[int] = set()
    for name, (values, expected_count) in embeddings.items():
        if values is None:
            continue
        if values.ndim != 2 or values.shape[0] != expected_count:
            raise ValueError(f"Backend {name} has an invalid shape; expected {expected_count} word vectors")
        if not np.isfinite(values).all():
            raise ValueError(f"Backend {name} contains a non-finite value")
        embedding_dimensions.add(int(values.shape[1]))
    if len(embedding_dimensions) > 1:
        raise ValueError("Backend source and target embeddings use different dimensions")


def _score_model_link(
    result: TokenEmbeddingAlignment,
    link: tuple[int, int],
    normalized_alignments: Mapping[str, set[tuple[int, int]]],
) -> LinkScore:
    source_index, target_index = link
    similarity = result.similarities[source_index][target_index]
    evidence_matrix = result.relative_similarities or result.similarities
    source_probability = result.source_to_target_probabilities[source_index][target_index]
    target_probability = result.target_to_source_probabilities[source_index][target_index]
    directional_evidence = math.sqrt(source_probability * target_probability)

    source_margin = _candidate_margin(
        evidence_matrix[source_index],
        selected_index=target_index,
    )
    target_column = tuple(row[target_index] for row in evidence_matrix)
    target_margin = _candidate_margin(target_column, selected_index=source_index)
    margin = (source_margin + target_margin) / 2.0

    enabled_methods = tuple(normalized_alignments.values())
    method_agreement = (
        sum(link in method_links for method_links in enabled_methods) / len(enabled_methods) if enabled_methods else 0.0
    )
    span_consistency = _link_span_consistency(
        link,
        normalized_alignments,
        source_count=len(result.similarities),
        target_count=len(result.similarities[0]) if result.similarities else 0,
    )
    confidence = 0.35 * directional_evidence + 0.30 * margin + 0.20 * method_agreement + 0.15 * span_consistency
    return LinkScore(
        similarity=_bounded_score(similarity),
        confidence=_bounded_score(confidence),
    )


def _candidate_margin(scores: Sequence[float], *, selected_index: int) -> float:
    competitors = [score for index, score in enumerate(scores) if index != selected_index]
    if not competitors:
        return 0.0
    return max(0.0, float(scores[selected_index]) - max(competitors))


def _link_span_consistency(
    link: tuple[int, int],
    normalized_alignments: Mapping[str, set[tuple[int, int]]],
    *,
    source_count: int,
    target_count: int,
) -> float:
    source_index, target_index = link
    order_scores: list[float] = []
    for method_links in normalized_alignments.values():
        if link not in method_links:
            continue
        comparable_links = [
            candidate
            for candidate in method_links
            if candidate != link and candidate[0] != source_index and candidate[1] != target_index
        ]
        if not comparable_links:
            order_scores.append(0.5)
            continue
        concordant = sum(
            (candidate_source - source_index) * (candidate_target - target_index) > 0
            for candidate_source, candidate_target in comparable_links
        )
        order_scores.append(concordant / len(comparable_links))

    order_consistency = sum(order_scores) / len(order_scores) if order_scores else 0.0
    position_consistency = 1.0 - abs(
        _relative_position(source_index, source_count) - _relative_position(target_index, target_count)
    )
    return _bounded_score(0.7 * order_consistency + 0.3 * position_consistency)


def _bounded_score(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)


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
    embedding_result: TokenEmbeddingAlignment,
    normalized_alignments: Mapping[str, set[tuple[int, int]]],
    max_position_distance: float,
    min_similarity: float,
    min_confidence: float,
) -> dict[tuple[int, int], LinkScore]:
    aligned_source = {source_index for source_index, _ in base_links}
    aligned_target = {target_index for _, target_index in base_links}
    candidates: list[tuple[float, float, float, int, int]] = []
    candidate_scores: dict[tuple[int, int], LinkScore] = {}

    for source_index, target_index in fallback_links:
        if source_index in aligned_source or target_index in aligned_target:
            continue
        if not _is_repairable_token(source_tokens[source_index]):
            continue
        if not _is_repairable_token(target_tokens[target_index]):
            continue

        position_distance = abs(
            _relative_position(source_index, len(source_tokens)) - _relative_position(target_index, len(target_tokens))
        )
        if position_distance > max_position_distance:
            continue

        model_score = _score_model_link(
            embedding_result,
            (source_index, target_index),
            normalized_alignments,
        )
        if model_score.similarity < min_similarity:
            continue

        position_score = _position_proximity(position_distance, max_position_distance)
        anchor_score = _anchor_consistency(source_index, target_index, base_links)
        repaired_confidence = model_score.confidence * (0.75 + 0.15 * position_score + 0.10 * anchor_score)
        repaired_score = LinkScore(
            similarity=model_score.similarity,
            confidence=_bounded_score(repaired_confidence),
        )
        if repaired_score.confidence < min_confidence:
            continue

        link = (source_index, target_index)
        candidate_scores[link] = repaired_score
        candidates.append(
            (
                -repaired_score.confidence,
                -repaired_score.similarity,
                position_distance,
                source_index,
                target_index,
            )
        )

    repaired_links: dict[tuple[int, int], LinkScore] = {}
    repaired_source: set[int] = set()
    repaired_target: set[int] = set()
    for _, _, _, source_index, target_index in sorted(candidates):
        if source_index in repaired_source or target_index in repaired_target:
            continue
        link = (source_index, target_index)
        repaired_links[link] = candidate_scores[link]
        repaired_source.add(source_index)
        repaired_target.add(target_index)

    return repaired_links


def _position_proximity(position_distance: float, max_position_distance: float) -> float:
    if max_position_distance == 0.0:
        return 1.0 if position_distance == 0.0 else 0.0
    return max(0.0, 1.0 - position_distance / max_position_distance)


def _anchor_consistency(
    source_index: int,
    target_index: int,
    base_links: set[tuple[int, int]],
) -> float:
    previous_source_indices = [
        candidate_source for candidate_source, _ in base_links if candidate_source < source_index
    ]
    next_source_indices = [candidate_source for candidate_source, _ in base_links if candidate_source > source_index]
    checks: list[bool] = []

    if previous_source_indices:
        previous_source = max(previous_source_indices)
        previous_targets = [
            candidate_target for candidate_source, candidate_target in base_links if candidate_source == previous_source
        ]
        checks.append(target_index >= max(previous_targets))

    if next_source_indices:
        next_source = min(next_source_indices)
        next_targets = [
            candidate_target for candidate_source, candidate_target in base_links if candidate_source == next_source
        ]
        checks.append(target_index <= min(next_targets))

    if not checks:
        return 0.5
    return sum(checks) / len(checks)


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
    link_scores: Mapping[tuple[int, int], LinkScore],
    refined_spans: Sequence[RefinedSpan],
    grouped_links: set[tuple[int, int]],
) -> SentenceAlignment:
    sorted_links = sorted(links)
    groups = _build_alignment_groups(
        source_tokens,
        target_tokens,
        [link for link in sorted_links if link not in grouped_links],
        link_origins,
        link_scores,
        refined_spans,
    )
    aligned_source = {source_index for source_index, _ in sorted_links}
    aligned_target = {target_index for _, target_index in sorted_links}
    aligned_source.update(source_index for span in refined_spans for source_index in span.source_indices)
    aligned_target.update(target_index for span in refined_spans for target_index in span.target_indices)

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
                similarity=link_scores[(source_index, target_index)].similarity,
                confidence=link_scores[(source_index, target_index)].confidence,
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
    link_scores: Mapping[tuple[int, int], LinkScore],
    refined_spans: Sequence[RefinedSpan],
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
                similarity=link_scores[(source_index, target_index)].similarity,
                confidence=link_scores[(source_index, target_index)].confidence,
            )
            for source_index, target_index in links
            if source_index in source_indices and target_index in target_indices
        ]
        groups.append(
            AlignmentGroup(
                type=_classify_group(len(source_indices), len(target_indices)),
                origin=_group_origin(component_links),
                similarity=_mean_group_score(component_links, "similarity"),
                confidence=_mean_group_score(component_links, "confidence"),
                source_indices=source_indices,
                target_indices=target_indices,
                source_tokens=[source_tokens[index].text for index in source_indices],
                target_tokens=[target_tokens[index].text for index in target_indices],
                links=component_links,
            )
        )
    for span in refined_spans:
        span_links = [
            AlignmentLink(
                source_index=source_index,
                target_index=target_index,
                origin=link_origins[(source_index, target_index)],
                similarity=link_scores[(source_index, target_index)].similarity,
                confidence=link_scores[(source_index, target_index)].confidence,
            )
            for source_index, target_index in span.links
        ]
        groups.append(
            AlignmentGroup(
                type=_classify_group(len(span.source_indices), len(span.target_indices)),
                origin="refined",
                similarity=span.similarity,
                confidence=span.confidence,
                source_indices=list(span.source_indices),
                target_indices=list(span.target_indices),
                source_tokens=[source_tokens[index].text for index in span.source_indices],
                target_tokens=[target_tokens[index].text for index in span.target_indices],
                links=span_links,
            )
        )
    groups.sort(
        key=lambda group: (
            group.source_indices[0] if group.source_indices else len(source_tokens),
            group.target_indices[0] if group.target_indices else len(target_tokens),
        )
    )
    return groups


def _group_origin(links: Sequence[AlignmentLink]) -> AlignmentGroupOrigin:
    origins = {link.origin for link in links}
    if len(origins) == 1:
        return next(iter(origins))
    return "mixed"


def _mean_group_score(links: Sequence[AlignmentLink], attribute: str) -> float:
    if not links:
        return 0.0
    return _bounded_score(sum(float(getattr(link, attribute)) for link in links) / len(links))


def _classify_group(source_count: int, target_count: int) -> str:
    if source_count == 1 and target_count == 1:
        return "one-to-one"
    if source_count == 1:
        return "one-to-many"
    if target_count == 1:
        return "many-to-one"
    return "many-to-many"
