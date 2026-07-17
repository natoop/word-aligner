from __future__ import annotations

import math
import unicodedata
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from app.schemas import Token

RawLink = tuple[int, int]
ScoreMatrix = Sequence[Sequence[float]]
SpanIndices = tuple[tuple[int, ...], tuple[int, ...]]
MAX_LOCAL_SEGMENTATION_TOKENS = 64


@dataclass(frozen=True, slots=True)
class RefinedSpan:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    links: tuple[RawLink, ...]
    similarity: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SpanRefinement:
    links: frozenset[RawLink]
    groups: tuple[RefinedSpan, ...]
    grouped_links: frozenset[RawLink]


@dataclass(frozen=True, slots=True)
class _Anchor:
    source_index: int
    target_index: int
    is_actual: bool


@dataclass(frozen=True, slots=True)
class _ClauseRegion:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    anchor_support: float


@dataclass(frozen=True, slots=True)
class _ExpansionCandidate:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    links: tuple[RawLink, ...]
    similarity: float
    confidence: float
    gain: float


@dataclass(frozen=True, slots=True)
class _PartitionState:
    score: float
    spans: tuple[SpanIndices, ...]


def refine_alignment_spans(
    *,
    source_tokens: Sequence[Token],
    target_tokens: Sequence[Token],
    links: set[RawLink],
    similarities: ScoreMatrix,
    relative_similarities: ScoreMatrix,
    link_confidences: Mapping[RawLink, float],
    strategy: str,
    max_source_span: int,
    max_target_span: int,
    min_score_gain: float,
    min_span_coverage: float,
    source_embeddings: np.ndarray | None = None,
    target_embeddings: np.ndarray | None = None,
) -> SpanRefinement:
    """Refine low-coverage clauses without inventing token-level correspondences."""

    current_links = set(links)
    regions = _extract_clause_regions(source_tokens, target_tokens, current_links)
    groups: list[RefinedSpan] = []
    refined_regions: set[SpanIndices] = set()

    for region in regions:
        region_links = _links_inside_region(current_links, region)
        has_low_coverage = _should_refine_region(
            region,
            region_links,
            current_links,
            min_span_coverage=min_span_coverage,
        )
        crossing_spans = _find_local_crossing_spans(
            region,
            region_links,
            max_source_span=max_source_span,
            max_target_span=max_target_span,
        )
        if not has_low_coverage and not crossing_spans:
            continue

        if has_low_coverage:
            partition = _partition_region(
                region,
                similarities,
                relative_similarities,
                source_embeddings,
                target_embeddings,
                max_source_span=max_source_span,
                max_target_span=max_target_span,
            )
            if partition is None:
                continue
            spans_to_refine = [
                span for span in partition if not _is_stable_partition_span(span[0], span[1], region_links)
            ]
        else:
            spans_to_refine = list(crossing_spans)
        if not spans_to_refine:
            continue

        groups.extend(
            _make_refined_span(
                source_indices,
                target_indices,
                region=region,
                region_links=region_links,
                similarities=similarities,
                relative_similarities=relative_similarities,
                link_confidences=link_confidences,
                source_embeddings=source_embeddings,
                target_embeddings=target_embeddings,
            )
            for source_indices, target_indices in spans_to_refine
        )
        refined_source = {source_index for source_indices, _ in spans_to_refine for source_index in source_indices}
        refined_target = {target_index for _, target_indices in spans_to_refine for target_index in target_indices}
        current_links = {
            link for link in current_links if link[0] not in refined_source and link[1] not in refined_target
        }
        refined_regions.add((region.source_indices, region.target_indices))

    grouped_links: set[RawLink] = set()
    if strategy == "span-aware":
        expansions = _find_span_expansions(
            regions=[
                region for region in regions if (region.source_indices, region.target_indices) not in refined_regions
            ],
            links=current_links,
            similarities=similarities,
            relative_similarities=relative_similarities,
            link_confidences=link_confidences,
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
            max_source_span=max_source_span,
            max_target_span=max_target_span,
            min_score_gain=min_score_gain,
            min_span_coverage=min_span_coverage,
        )
        groups.extend(expansions)
        grouped_links.update(link for expansion in expansions for link in expansion.links)

    groups.sort(key=lambda group: (group.source_indices[0], group.target_indices[0]))
    return SpanRefinement(
        links=frozenset(current_links),
        groups=tuple(groups),
        grouped_links=frozenset(grouped_links),
    )


def _extract_clause_regions(
    source_tokens: Sequence[Token],
    target_tokens: Sequence[Token],
    links: set[RawLink],
) -> list[_ClauseRegion]:
    anchor_candidates = sorted(
        (
            _Anchor(source_index, target_index, True)
            for source_index, target_index in links
            if _is_anchor_pair(source_tokens[source_index], target_tokens[target_index])
        ),
        key=lambda anchor: (anchor.source_index, anchor.target_index),
    )
    anchors = [_Anchor(-1, -1, False)]
    for candidate in anchor_candidates:
        previous = anchors[-1]
        if candidate.source_index > previous.source_index and candidate.target_index > previous.target_index:
            anchors.append(candidate)
    anchors.append(_Anchor(len(source_tokens), len(target_tokens), False))

    regions: list[_ClauseRegion] = []
    for previous, following in zip(anchors, anchors[1:], strict=False):
        source_indices = tuple(range(previous.source_index + 1, following.source_index))
        target_indices = tuple(range(previous.target_index + 1, following.target_index))
        if not source_indices or not target_indices:
            continue
        if not all(_is_content_token(source_tokens[index]) for index in source_indices):
            continue
        if not all(_is_content_token(target_tokens[index]) for index in target_indices):
            continue
        if not previous.is_actual and not following.is_actual:
            continue
        regions.append(
            _ClauseRegion(
                source_indices=source_indices,
                target_indices=target_indices,
                anchor_support=(float(previous.is_actual) + float(following.is_actual)) / 2.0,
            )
        )
    return regions


def _is_anchor_pair(source_token: Token, target_token: Token) -> bool:
    if source_token.is_protected or target_token.is_protected:
        return source_token.is_protected and target_token.is_protected and source_token.text == target_token.text
    return _is_punctuation_token(source_token) and _is_punctuation_token(target_token)


def _is_punctuation_token(token: Token) -> bool:
    return bool(token.text) and all(unicodedata.category(character).startswith("P") for character in token.text)


def _is_content_token(token: Token) -> bool:
    if token.is_protected:
        return False
    return any(unicodedata.category(character)[0] in {"L", "N"} for character in token.text)


def _links_inside_region(links: set[RawLink], region: _ClauseRegion) -> set[RawLink]:
    source_set = set(region.source_indices)
    target_set = set(region.target_indices)
    return {link for link in links if link[0] in source_set and link[1] in target_set}


def _should_refine_region(
    region: _ClauseRegion,
    region_links: set[RawLink],
    all_links: set[RawLink],
    *,
    min_span_coverage: float,
) -> bool:
    if len(region.source_indices) < 2:
        return False
    if len(region.target_indices) < 2:
        return False
    if len(region_links) < 2:
        return False

    source_set = set(region.source_indices)
    target_set = set(region.target_indices)
    if any((source_index in source_set) != (target_index in target_set) for source_index, target_index in all_links):
        return False

    source_coverage, target_coverage = _region_coverage(region, region_links)
    minimum_evidence_coverage = min(0.5, min_span_coverage)
    return (
        min(source_coverage, target_coverage) < min_span_coverage
        and max(source_coverage, target_coverage) >= minimum_evidence_coverage
    )


def _region_coverage(
    region: _ClauseRegion,
    region_links: set[RawLink],
) -> tuple[float, float]:
    aligned_source = {source_index for source_index, _ in region_links}
    aligned_target = {target_index for _, target_index in region_links}
    return (
        len(aligned_source) / len(region.source_indices),
        len(aligned_target) / len(region.target_indices),
    )


def _find_local_crossing_spans(
    region: _ClauseRegion,
    region_links: set[RawLink],
    *,
    max_source_span: int,
    max_target_span: int,
) -> tuple[SpanIndices, ...]:
    if max_source_span < 2 or max_target_span < 2:
        return ()

    source_targets: dict[int, set[int]] = defaultdict(set)
    target_sources: dict[int, set[int]] = defaultdict(set)
    for source_index, target_index in region_links:
        source_targets[source_index].add(target_index)
        target_sources[target_index].add(source_index)

    unique_links = {
        source_index: next(iter(targets))
        for source_index, targets in source_targets.items()
        if len(targets) == 1 and len(target_sources[next(iter(targets))]) == 1
    }
    source_indices = sorted(unique_links)
    spans: list[SpanIndices] = []
    cursor = 0
    while cursor < len(source_indices):
        run_sources = [source_indices[cursor]]
        run_targets = [unique_links[source_indices[cursor]]]
        next_cursor = cursor + 1
        while next_cursor < len(source_indices):
            next_source = source_indices[next_cursor]
            next_target = unique_links[next_source]
            if next_source != run_sources[-1] + 1 or next_target != run_targets[-1] - 1:
                break
            if len(run_sources) >= min(max_source_span, max_target_span):
                break
            run_sources.append(next_source)
            run_targets.append(next_target)
            next_cursor += 1

        if len(run_sources) >= 2:
            target_indices = tuple(sorted(run_targets))
            if (
                tuple(run_sources) == tuple(range(run_sources[0], run_sources[-1] + 1))
                and target_indices == tuple(range(target_indices[0], target_indices[-1] + 1))
                and set(run_sources).issubset(region.source_indices)
                and set(target_indices).issubset(region.target_indices)
            ):
                spans.append((tuple(run_sources), target_indices))
                cursor = next_cursor
                continue
        cursor += 1
    return tuple(spans)


def _partition_region(
    region: _ClauseRegion,
    similarities: ScoreMatrix,
    relative_similarities: ScoreMatrix,
    source_embeddings: np.ndarray | None,
    target_embeddings: np.ndarray | None,
    *,
    max_source_span: int,
    max_target_span: int,
) -> tuple[SpanIndices, ...] | None:
    if len(region.source_indices) <= max_source_span and len(region.target_indices) <= max_target_span:
        return ((region.source_indices, region.target_indices),)
    if len(region.source_indices) + len(region.target_indices) > MAX_LOCAL_SEGMENTATION_TOKENS:
        return None

    source_count = len(region.source_indices)
    target_count = len(region.target_indices)
    states: dict[tuple[int, int], _PartitionState] = {(0, 0): _PartitionState(score=0.0, spans=())}
    for source_offset in range(source_count + 1):
        for target_offset in range(target_count + 1):
            state = states.get((source_offset, target_offset))
            if state is None:
                continue
            for source_length in range(1, max_source_span + 1):
                source_end = source_offset + source_length
                if source_end > source_count:
                    break
                for target_length in range(1, max_target_span + 1):
                    target_end = target_offset + target_length
                    if target_end > target_count:
                        break
                    if source_length > 1 and target_length > 1:
                        continue

                    source_indices = region.source_indices[source_offset:source_end]
                    target_indices = region.target_indices[target_offset:target_end]
                    semantic_score = _span_similarity(
                        source_indices,
                        target_indices,
                        similarities,
                        source_embeddings,
                        target_embeddings,
                    )
                    relative_score = _matrix_span_score(
                        source_indices,
                        target_indices,
                        relative_similarities,
                    )
                    boundary_score = _balanced_boundary_score(
                        source_end,
                        target_end,
                        source_count,
                        target_count,
                    )
                    candidate_score = 0.45 * semantic_score + 0.10 * relative_score + 0.45 * boundary_score
                    candidate_state = _PartitionState(
                        score=state.score + candidate_score,
                        spans=state.spans + ((source_indices, target_indices),),
                    )
                    key = (source_end, target_end)
                    existing = states.get(key)
                    if _is_better_partition(candidate_state, existing):
                        states[key] = candidate_state

    result = states.get((source_count, target_count))
    return result.spans if result is not None else None


def _balanced_boundary_score(
    source_end: int,
    target_end: int,
    source_count: int,
    target_count: int,
) -> float:
    if source_count >= target_count:
        preferred_source_end = math.floor(target_end * source_count / target_count + 0.5)
        distance = abs(source_end - preferred_source_end)
    else:
        preferred_target_end = math.floor(source_end * target_count / source_count + 0.5)
        distance = abs(target_end - preferred_target_end)
    return max(0.0, 1.0 - distance / max(source_count, target_count))


def _is_better_partition(
    candidate: _PartitionState,
    existing: _PartitionState | None,
) -> bool:
    if existing is None:
        return True
    if not math.isclose(candidate.score, existing.score, abs_tol=1e-12):
        return candidate.score > existing.score
    if len(candidate.spans) != len(existing.spans):
        return len(candidate.spans) > len(existing.spans)
    candidate_lengths = tuple(
        (len(source_indices), len(target_indices)) for source_indices, target_indices in candidate.spans
    )
    existing_lengths = tuple(
        (len(source_indices), len(target_indices)) for source_indices, target_indices in existing.spans
    )
    return candidate_lengths > existing_lengths


def _is_stable_partition_span(
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    region_links: set[RawLink],
) -> bool:
    source_set = set(source_indices)
    target_set = set(target_indices)
    incident_links = {link for link in region_links if link[0] in source_set or link[1] in target_set}
    internal_links = {link for link in incident_links if link[0] in source_set and link[1] in target_set}
    return (
        bool(internal_links)
        and incident_links == internal_links
        and {source_index for source_index, _ in internal_links} == source_set
        and {target_index for _, target_index in internal_links} == target_set
    )


def _make_refined_span(
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    *,
    region: _ClauseRegion,
    region_links: set[RawLink],
    similarities: ScoreMatrix,
    relative_similarities: ScoreMatrix,
    link_confidences: Mapping[RawLink, float],
    source_embeddings: np.ndarray | None,
    target_embeddings: np.ndarray | None,
) -> RefinedSpan:
    source_set = set(source_indices)
    target_set = set(target_indices)
    evidence_links = {link for link in region_links if link[0] in source_set or link[1] in target_set}
    similarity = _span_similarity(
        source_indices,
        target_indices,
        similarities,
        source_embeddings,
        target_embeddings,
    )
    relative_score = _matrix_span_score(
        source_indices,
        target_indices,
        relative_similarities,
    )
    confidence = _bounded_score(
        0.40 * relative_score
        + 0.25 * _mean_link_confidence(evidence_links, link_confidences)
        + 0.20 * similarity
        + 0.15 * region.anchor_support
    )
    return RefinedSpan(
        source_indices=source_indices,
        target_indices=target_indices,
        links=(),
        similarity=similarity,
        confidence=confidence,
    )


def _find_span_expansions(
    *,
    regions: Sequence[_ClauseRegion],
    links: set[RawLink],
    similarities: ScoreMatrix,
    relative_similarities: ScoreMatrix,
    link_confidences: Mapping[RawLink, float],
    source_embeddings: np.ndarray | None,
    target_embeddings: np.ndarray | None,
    max_source_span: int,
    max_target_span: int,
    min_score_gain: float,
    min_span_coverage: float,
) -> list[RefinedSpan]:
    aligned_source = {source_index for source_index, _ in links}
    aligned_target = {target_index for _, target_index in links}
    candidates: list[_ExpansionCandidate] = []

    for region in regions:
        region_links = _links_inside_region(links, region)
        for component_links in _link_components(region_links):
            source_indices = tuple(sorted({source_index for source_index, _ in component_links}))
            target_indices = tuple(sorted({target_index for _, target_index in component_links}))
            if not _is_contiguous(source_indices) or not _is_contiguous(target_indices):
                continue

            base_similarity = _span_similarity(
                source_indices,
                target_indices,
                similarities,
                source_embeddings,
                target_embeddings,
            )
            mean_link_confidence = _mean_link_confidence(component_links, link_confidences)

            for expanded_source in _containing_ranges(
                source_indices,
                region.source_indices,
                max_source_span,
                aligned_source,
            ):
                if len(target_indices) > max_target_span:
                    continue
                coverage = len((aligned_source & set(region.source_indices)) | set(expanded_source)) / len(
                    region.source_indices
                )
                candidate = _make_expansion_candidate(
                    source_indices=expanded_source,
                    target_indices=target_indices,
                    links=component_links,
                    base_similarity=base_similarity,
                    coverage=coverage,
                    min_span_coverage=min_span_coverage,
                    min_score_gain=min_score_gain,
                    similarities=similarities,
                    relative_similarities=relative_similarities,
                    mean_link_confidence=mean_link_confidence,
                    source_embeddings=source_embeddings,
                    target_embeddings=target_embeddings,
                )
                if candidate is not None:
                    candidates.append(candidate)

            for expanded_target in _containing_ranges(
                target_indices,
                region.target_indices,
                max_target_span,
                aligned_target,
            ):
                if len(source_indices) > max_source_span:
                    continue
                coverage = len((aligned_target & set(region.target_indices)) | set(expanded_target)) / len(
                    region.target_indices
                )
                candidate = _make_expansion_candidate(
                    source_indices=source_indices,
                    target_indices=expanded_target,
                    links=component_links,
                    base_similarity=base_similarity,
                    coverage=coverage,
                    min_span_coverage=min_span_coverage,
                    min_score_gain=min_score_gain,
                    similarities=similarities,
                    relative_similarities=relative_similarities,
                    mean_link_confidence=mean_link_confidence,
                    source_embeddings=source_embeddings,
                    target_embeddings=target_embeddings,
                )
                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -candidate.gain,
            -candidate.confidence,
            len(candidate.source_indices) + len(candidate.target_indices),
            candidate.source_indices[0],
            candidate.target_indices[0],
        )
    )
    claimed_source: set[int] = set()
    claimed_target: set[int] = set()
    claimed_links: set[RawLink] = set()
    expansions: list[RefinedSpan] = []
    for candidate in candidates:
        if claimed_source.intersection(candidate.source_indices):
            continue
        if claimed_target.intersection(candidate.target_indices):
            continue
        if claimed_links.intersection(candidate.links):
            continue
        claimed_source.update(candidate.source_indices)
        claimed_target.update(candidate.target_indices)
        claimed_links.update(candidate.links)
        expansions.append(
            RefinedSpan(
                source_indices=candidate.source_indices,
                target_indices=candidate.target_indices,
                links=candidate.links,
                similarity=candidate.similarity,
                confidence=candidate.confidence,
            )
        )
    return expansions


def _make_expansion_candidate(
    *,
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    links: set[RawLink],
    base_similarity: float,
    coverage: float,
    min_span_coverage: float,
    min_score_gain: float,
    similarities: ScoreMatrix,
    relative_similarities: ScoreMatrix,
    mean_link_confidence: float,
    source_embeddings: np.ndarray | None,
    target_embeddings: np.ndarray | None,
) -> _ExpansionCandidate | None:
    if coverage < min_span_coverage:
        return None
    similarity = _span_similarity(
        source_indices,
        target_indices,
        similarities,
        source_embeddings,
        target_embeddings,
    )
    gain = similarity - base_similarity
    if gain < min_score_gain:
        return None
    relative_score = _matrix_span_score(
        source_indices,
        target_indices,
        relative_similarities,
    )
    confidence = _bounded_score(
        0.45 * relative_score
        + 0.30 * mean_link_confidence
        + 0.15 * coverage
        + 0.10 * min(1.0, gain / max(min_score_gain, 1e-12))
    )
    return _ExpansionCandidate(
        source_indices=source_indices,
        target_indices=target_indices,
        links=tuple(sorted(links)),
        similarity=similarity,
        confidence=confidence,
        gain=gain,
    )


def _containing_ranges(
    base_indices: tuple[int, ...],
    region_indices: tuple[int, ...],
    max_span: int,
    aligned_indices: set[int],
) -> list[tuple[int, ...]]:
    region_start = region_indices[0]
    region_end = region_indices[-1]
    base_start = base_indices[0]
    base_end = base_indices[-1]
    base_set = set(base_indices)
    ranges: list[tuple[int, ...]] = []
    for start in range(region_start, base_start + 1):
        for end in range(base_end, region_end + 1):
            candidate = tuple(range(start, end + 1))
            if len(candidate) > max_span or set(candidate) == base_set:
                continue
            added = set(candidate) - base_set
            if added.intersection(aligned_indices):
                continue
            ranges.append(candidate)
    return ranges


def _link_components(links: set[RawLink]) -> list[set[RawLink]]:
    by_source: dict[int, set[RawLink]] = defaultdict(set)
    by_target: dict[int, set[RawLink]] = defaultdict(set)
    for link in links:
        by_source[link[0]].add(link)
        by_target[link[1]].add(link)

    components: list[set[RawLink]] = []
    visited: set[RawLink] = set()
    for link in sorted(links):
        if link in visited:
            continue
        queue = deque([link])
        component: set[RawLink] = set()
        visited.add(link)
        while queue:
            current = queue.popleft()
            component.add(current)
            neighbors = by_source[current[0]] | by_target[current[1]]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _is_contiguous(indices: tuple[int, ...]) -> bool:
    return indices == tuple(range(indices[0], indices[-1] + 1))


def _span_similarity(
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    similarities: ScoreMatrix,
    source_embeddings: np.ndarray | None,
    target_embeddings: np.ndarray | None,
) -> float:
    if source_embeddings is not None and target_embeddings is not None:
        source_vector = np.mean(source_embeddings[list(source_indices)], axis=0)
        target_vector = np.mean(target_embeddings[list(target_indices)], axis=0)
        denominator = float(np.linalg.norm(source_vector) * np.linalg.norm(target_vector))
        if denominator > 0.0:
            cosine = float(np.dot(source_vector, target_vector) / denominator)
            return _bounded_score((cosine + 1.0) / 2.0)
    return _bounded_score(_matrix_span_score(source_indices, target_indices, similarities))


def _matrix_span_score(
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    matrix: ScoreMatrix,
) -> float:
    source_scores = [
        max(float(matrix[source_index][target_index]) for target_index in target_indices)
        for source_index in source_indices
    ]
    target_scores = [
        max(float(matrix[source_index][target_index]) for source_index in source_indices)
        for target_index in target_indices
    ]
    forward = sum(source_scores) / len(source_scores)
    reverse = sum(target_scores) / len(target_scores)
    return math.sqrt(max(0.0, forward * reverse))


def _mean_link_confidence(
    links: set[RawLink],
    link_confidences: Mapping[RawLink, float],
) -> float:
    if not links:
        return 0.0
    return sum(link_confidences.get(link, 0.0) for link in links) / len(links)


def _bounded_score(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 6)
