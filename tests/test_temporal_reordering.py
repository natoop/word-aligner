from __future__ import annotations

from collections.abc import Callable

import pytest

from app.schemas import Token
from app.temporal_reordering import refine_temporal_reordering

SpanSimilarity = Callable[[tuple[int, ...], tuple[int, ...]], float]


def _tokens(words: tuple[str, ...]) -> list[Token]:
    return [Token(index=index, text=word, start=index, end=index + len(word)) for index, word in enumerate(words)]


def _standard_case() -> tuple[list[Token], list[Token], tuple[int, ...], tuple[int, ...]]:
    source = _tokens(
        (
            "Callback",
            "triggered",
            "before",
            "the",
            "cloned",
            "node",
            "is",
            "inserted",
            "into",
            "the",
            "original",
            "tree",
        )
    )
    target = _tokens(("克隆", "节点", "插入", "原树", "之前", "触发", "的", "回调"))
    return source, target, tuple(range(len(source))), tuple(range(len(target)))


def _semantic_similarity(source: list[Token], target: list[Token]) -> SpanSimilarity:
    matches = {
        (("Callback",), ("回调",)): 0.99,
        (("triggered",), ("触发",)): 0.98,
        (("triggered",), ("触发", "的")): 0.99,
        (("before",), ("之前",)): 0.99,
        (("cloned",), ("克隆",)): 0.99,
        (("node",), ("节点",)): 0.99,
        (("inserted",), ("插入",)): 0.99,
        (("original", "tree"), ("原树",)): 0.99,
    }

    def score(source_indices: tuple[int, ...], target_indices: tuple[int, ...]) -> float:
        source_words = tuple(source[index].text for index in source_indices)
        target_words = tuple(target[index].text for index in target_indices)
        return matches.get((source_words, target_words), matches.get((target_words, source_words), 0.1))

    return score


def _run(
    source: list[Token],
    target: list[Token],
    *,
    span_similarity: SpanSimilarity,
    max_source_span: int = 3,
    max_target_span: int = 6,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return refine_temporal_reordering(
        source,
        target,
        tuple(range(len(source))),
        tuple(range(len(target))),
        span_similarity=span_similarity,
        min_similarity=0.45,
        max_source_span=max_source_span,
        max_target_span=max_target_span,
    )


def test_temporal_reordering_returns_empty_without_a_matching_marker() -> None:
    source = _tokens(("Callback", "triggered", "while", "the", "cloned", "node", "is", "inserted"))
    target = _tokens(("克隆", "节点", "插入", "原树", "时", "触发", "的", "回调"))

    result = _run(source, target, span_similarity=lambda _source, _target: 0.99)

    assert result == ()


def test_temporal_reordering_returns_empty_when_the_marker_is_repeated() -> None:
    source, target, _, _ = _standard_case()
    target.append(Token(index=len(target), text="之前", start=len(target), end=len(target) + 2))

    result = _run(source, target, span_similarity=_semantic_similarity(source, target))

    assert result == ()


def test_temporal_reordering_returns_empty_when_marker_similarity_is_too_low() -> None:
    source, target, _, _ = _standard_case()

    def score(source_indices: tuple[int, ...], target_indices: tuple[int, ...]) -> float:
        source_words = tuple(source[index].text for index in source_indices)
        target_words = tuple(target[index].text for index in target_indices)
        if (source_words, target_words) in {(("before",), ("之前",)), (("之前",), ("before",))}:
            return 0.44
        return 0.99

    result = _run(source, target, span_similarity=score)

    assert result == ()


@pytest.mark.parametrize(
    ("max_source_span", "max_target_span"),
    [(1, 6), (3, 1)],
)
def test_temporal_reordering_returns_empty_when_span_limits_cannot_cover_blocks(
    max_source_span: int,
    max_target_span: int,
) -> None:
    source, target, _, _ = _standard_case()

    result = _run(
        source,
        target,
        span_similarity=_semantic_similarity(source, target),
        max_source_span=max_source_span,
        max_target_span=max_target_span,
    )

    assert result == ()


def test_temporal_reordering_returns_empty_when_a_clause_side_has_fewer_than_two_words() -> None:
    source = _tokens(("Callback", "before", "the", "cloned", "node"))
    target = _tokens(("克隆", "节点", "之前", "回调"))

    result = _run(source, target, span_similarity=lambda _source, _target: 0.99)

    assert result == ()
