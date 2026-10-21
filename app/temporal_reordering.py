from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.schemas import Token

SpanIndices = tuple[tuple[int, ...], tuple[int, ...]]
SpanSimilarity = Callable[[tuple[int, ...], tuple[int, ...]], float]

# These are grammatical relation markers, not a translation dictionary for
# content words. Their position determines which clauses can change order.
_TEMPORAL_MARKERS = {
    "before": frozenset({"之前", "以前"}),
    "after": frozenset({"之后", "以后"}),
}
_ENGLISH_NULL_WORDS = frozenset({"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "into", "to"})
_CHINESE_FUNCTION_WORDS = frozenset({"的"})


@dataclass(frozen=True, slots=True)
class _Path:
    score: float
    spans: tuple[SpanIndices, ...]


def refine_temporal_reordering(
    source_tokens: Sequence[Token],
    target_tokens: Sequence[Token],
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    *,
    span_similarity: SpanSimilarity,
    min_similarity: float,
    max_source_span: int,
    max_target_span: int,
) -> tuple[SpanIndices, ...]:
    """Align lexical units inside an English/Chinese temporal clause swap."""

    forward = _refine_english_chinese(
        source_tokens,
        target_tokens,
        source_indices,
        target_indices,
        span_similarity=span_similarity,
        min_similarity=min_similarity,
        max_english_span=max_source_span,
        max_chinese_span=max_target_span,
    )
    if forward:
        return forward
    reverse = _refine_english_chinese(
        target_tokens,
        source_tokens,
        target_indices,
        source_indices,
        span_similarity=lambda english, chinese: span_similarity(chinese, english),
        min_similarity=min_similarity,
        max_english_span=max_target_span,
        max_chinese_span=max_source_span,
    )
    return tuple((chinese, english) for english, chinese in reverse)


def _refine_english_chinese(
    english_tokens: Sequence[Token],
    chinese_tokens: Sequence[Token],
    english_indices: tuple[int, ...],
    chinese_indices: tuple[int, ...],
    *,
    span_similarity: SpanSimilarity,
    min_similarity: float,
    max_english_span: int,
    max_chinese_span: int,
) -> tuple[SpanIndices, ...]:
    markers = [
        (english_index, chinese_index)
        for english_index in english_indices
        if english_tokens[english_index].text.casefold() in _TEMPORAL_MARKERS
        for chinese_index in chinese_indices
        if chinese_tokens[chinese_index].text in _TEMPORAL_MARKERS[english_tokens[english_index].text.casefold()]
    ]
    if len(markers) != 1:
        return ()
    english_marker, chinese_marker = markers[0]
    english_left = tuple(
        index
        for index in english_indices
        if index < english_marker and english_tokens[index].text.casefold() not in _ENGLISH_NULL_WORDS
    )
    english_right = tuple(
        index
        for index in english_indices
        if index > english_marker and english_tokens[index].text.casefold() not in _ENGLISH_NULL_WORDS
    )
    chinese_left = tuple(index for index in chinese_indices if index < chinese_marker)
    chinese_right = tuple(index for index in chinese_indices if index > chinese_marker)
    if min(len(english_left), len(english_right), len(chinese_left), len(chinese_right)) < 2:
        return ()
    if span_similarity((english_marker,), (chinese_marker,)) < min_similarity:
        return ()

    left = _align_lexical_block(
        english_left,
        chinese_right,
        chinese_tokens,
        span_similarity=span_similarity,
        min_similarity=min_similarity,
        max_english_span=max_english_span,
        max_chinese_span=max_chinese_span,
        allow_reversal=True,
    )
    right = _align_lexical_block(
        english_right,
        chinese_left,
        chinese_tokens,
        span_similarity=span_similarity,
        min_similarity=min_similarity,
        max_english_span=max_english_span,
        max_chinese_span=max_chinese_span,
        allow_reversal=False,
    )
    if left is None or right is None:
        return ()
    return (*left.spans, ((english_marker,), (chinese_marker,)), *right.spans)


def _align_lexical_block(
    english_indices: tuple[int, ...],
    chinese_indices: tuple[int, ...],
    chinese_tokens: Sequence[Token],
    *,
    span_similarity: SpanSimilarity,
    min_similarity: float,
    max_english_span: int,
    max_chinese_span: int,
    allow_reversal: bool,
) -> _Path | None:
    orientations = [english_indices]
    chinese_content_count = sum(chinese_tokens[index].text not in _CHINESE_FUNCTION_WORDS for index in chinese_indices)
    if allow_reversal and len(english_indices) == chinese_content_count == 2:
        orientations.append(tuple(reversed(english_indices)))

    candidates: list[_Path] = []
    for english_order in orientations:
        states = {(0, 0): _Path(0.0, ())}
        for source_offset in range(len(english_order)):
            for target_offset in range(len(chinese_indices)):
                state = states.get((source_offset, target_offset))
                if state is None:
                    continue
                for source_length in range(1, min(2, max_english_span) + 1):
                    source_end = source_offset + source_length
                    source_span = tuple(sorted(english_order[source_offset:source_end]))
                    if len(source_span) != source_length or source_span[-1] - source_span[0] + 1 != source_length:
                        continue
                    for target_length in range(1, min(2, max_chinese_span) + 1):
                        target_end = target_offset + target_length
                        target_span = chinese_indices[target_offset:target_end]
                        if len(target_span) != target_length or source_length > 1 and target_length > 1:
                            continue
                        if sum(chinese_tokens[index].text not in _CHINESE_FUNCTION_WORDS for index in target_span) != 1:
                            continue
                        similarity = span_similarity(source_span, target_span)
                        if similarity < min_similarity:
                            continue
                        weight = source_length + target_length
                        score = state.score + weight * similarity - 0.01 * (weight - 2)
                        key = (source_end, target_end)
                        if key not in states or score > states[key].score:
                            states[key] = _Path(score, (*state.spans, (source_span, target_span)))
        result = states.get((len(english_order), len(chinese_indices)))
        if result is not None:
            candidates.append(result)
    return max(candidates, key=lambda candidate: candidate.score) if candidates else None
