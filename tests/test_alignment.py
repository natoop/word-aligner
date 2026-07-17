from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.alignment import (
    AlignmentModelError,
    AlignmentService,
    SimAlignBackend,
    TokenEmbeddingAlignment,
    _ensure_device_available,
)
from app.config import Settings
from app.schemas import AlignmentRequest
from app.tokenization import WordTokenizer


def _whitespace_tokenize(value: str):
    for match in re.finditer(r"\S+", value):
        yield match.group(), match.start(), match.end()


def _direction_phrase_tokenize(value: str):
    for match in re.finditer(r"北方|南方|的|[，。]", value):
        yield match.group(), match.start(), match.end()


def _machine_translation_tokenize(value: str):
    for match in re.finditer(r"这是|一个|机器翻译|示例|。", value):
        yield match.group(), match.start(), match.end()


def _atomic_link_tokenize(value: str):
    for match in re.finditer(r"原子|链接|只|保留|和", value):
        yield match.group(), match.start(), match.end()


class StubBackend:
    def __init__(
        self,
        links: set[tuple[int, int]],
        additional_alignments: dict[str, set[tuple[int, int]]] | None = None,
        similarities: list[list[float]] | None = None,
        source_embeddings: list[list[float]] | None = None,
        target_embeddings: list[list[float]] | None = None,
    ) -> None:
        self._alignments = {"itermax": links, **(additional_alignments or {})}
        self._similarities = similarities
        self._source_embeddings = source_embeddings
        self._target_embeddings = target_embeddings
        self.calls = 0

    def align_tokens(self, source_tokens: list[str], target_tokens: list[str]) -> TokenEmbeddingAlignment:
        self.calls += 1
        if self._similarities is None:
            matrix = np.full((len(source_tokens), len(target_tokens)), 0.1, dtype=float)
            for source_index, target_index in set().union(*self._alignments.values()):
                matrix[source_index, target_index] = 0.9
        else:
            matrix = np.asarray(self._similarities, dtype=float)

        source_probabilities = _softmax(matrix, axis=1)
        target_probabilities = _softmax(matrix, axis=0)
        return TokenEmbeddingAlignment(
            alignments=self._alignments,
            similarities=_freeze(matrix),
            source_to_target_probabilities=_freeze(source_probabilities),
            target_to_source_probabilities=_freeze(target_probabilities),
            source_embeddings=(
                np.asarray(self._source_embeddings, dtype=float) if self._source_embeddings is not None else None
            ),
            target_embeddings=(
                np.asarray(self._target_embeddings, dtype=float) if self._target_embeddings is not None else None
            ),
        )


def _softmax(matrix: np.ndarray, *, axis: int) -> np.ndarray:
    scaled = matrix / 0.1
    scaled -= scaled.max(axis=axis, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _freeze(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _request(source: str, target: str) -> AlignmentRequest:
    return AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "zh-Hans",
            "sentence_pairs": [{"id": "pair-1", "source": source, "target": target}],
        }
    )


def test_service_groups_many_to_one_and_one_to_many_links() -> None:
    backend = StubBackend({(0, 0), (1, 0), (2, 1), (2, 2), (3, 3)})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=lambda: backend,
    )

    response = service.align(_request("New York loves tea", "纽约 非常 喜欢 茶"))

    result = response.sentence_alignments[0]
    assert [group.type for group in result.alignment_groups] == ["many-to-one", "one-to-many", "one-to-one"]
    assert result.alignment_groups[0].source_tokens == ["New", "York"]
    assert result.alignment_groups[0].target_tokens == ["纽约"]
    assert result.alignment_groups[1].source_tokens == ["loves"]
    assert result.alignment_groups[1].target_tokens == ["非常", "喜欢"]
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []
    assert all(0.0 <= link.similarity <= 1.0 for link in result.links)
    assert all(0.0 <= link.confidence <= 1.0 for link in result.links)
    assert backend.calls == 1


def test_service_reports_unaligned_tokens() -> None:
    backend = StubBackend({(0, 0)})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=lambda: backend,
    )

    response = service.align(_request("hello extra", "你好 额外"))

    result = response.sentence_alignments[0]
    assert result.unaligned_source_indices == [1]
    assert result.unaligned_target_indices == [1]


def test_exact_placeholder_alignment_overrides_neural_links() -> None:
    backend = StubBackend({(0, 1), (1, 0)})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=lambda: backend,
    )

    response = service.align(_request("Keep [[T1504_1]] now", "保留 [[T1504_1]]"))

    result = response.sentence_alignments[0]
    assert [(link.source_index, link.target_index) for link in result.links] == [(1, 1)]
    assert result.links[0].origin == "rule"
    assert result.links[0].similarity == 1.0
    assert result.links[0].confidence == 1.0
    assert result.alignment_groups[0].source_tokens == ["[[T1504_1]]"]
    assert result.alignment_groups[0].target_tokens == ["[[T1504_1]]"]


def test_backend_is_loaded_only_once_for_multiple_requests() -> None:
    backend = StubBackend({(0, 0)})
    factory_calls = 0

    def factory() -> StubBackend:
        nonlocal factory_calls
        factory_calls += 1
        return backend

    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=factory,
    )

    service.align(_request("one", "一"))
    service.align(_request("two", "二"))

    assert factory_calls == 1
    assert backend.calls == 2


def test_service_preserves_input_whitespace_and_offsets() -> None:
    backend = StubBackend({(0, 0)})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=lambda: backend,
    )

    response = service.align(_request("  hello  ", "  你好  "))

    result = response.sentence_alignments[0]
    assert result.source == "  hello  "
    assert result.target == "  你好  "
    assert (result.source_tokens[0].start, result.source_tokens[0].end) == (2, 7)
    assert (result.target_tokens[0].start, result.target_tokens[0].end) == (2, 4)


def test_cuda_configuration_fails_with_an_actionable_error_when_cuda_is_unavailable() -> None:
    with pytest.raises(AlignmentModelError, match="Set ALIGNER_DEVICE=cpu"):
        _ensure_device_available("cuda", cuda_available=False)


def test_cpu_configuration_does_not_require_cuda() -> None:
    _ensure_device_available("cpu", cuda_available=False)


def test_backend_reports_missing_sentencepiece_dependency() -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with (
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("app.alignment.find_spec", return_value=None),
        pytest.raises(AlignmentModelError, match="required by the XLM-R tokenizer"),
    ):
        SimAlignBackend(Settings(device="cpu"))


def test_simalign_backend_reuses_one_embedding_inference_for_links_and_scores() -> None:
    class FakeTensor:
        def __init__(self, value: np.ndarray) -> None:
            self._value = value
            self.shape = value.shape

        def cpu(self) -> FakeTensor:
            return self

        def detach(self) -> FakeTensor:
            return self

        def numpy(self) -> np.ndarray:
            return self._value

    class FakeEmbeddingLoader:
        def __init__(self) -> None:
            self.tokenizer = SimpleNamespace(tokenize=lambda value: [value], unk_token="[UNK]")
            self.calls = 0

        def get_embed_list(self, _: list[list[str]]) -> FakeTensor:
            self.calls += 1
            return FakeTensor(
                np.asarray(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.0], [0.0, 1.0]],
                    ]
                )
            )

    embedding_loader = FakeEmbeddingLoader()
    identity = np.eye(2)
    fake_aligner = SimpleNamespace(
        embed_loader=embedding_loader,
        matching_methods=["inter", "mwmf", "itermax"],
        average_embeds_over_words=lambda vectors, _: vectors,
        get_similarity=lambda source, target: np.matmul(source, target.transpose()),
        get_alignment_matrix=lambda _: (identity, identity),
        get_max_weight_match=lambda _: identity,
        iter_max=lambda _: identity,
    )
    backend = SimAlignBackend.__new__(SimAlignBackend)
    backend._aligner = fake_aligner
    backend._confidence_temperature = 0.1

    result = backend.align_tokens(["one", "two"], ["一", "二"])

    assert embedding_loader.calls == 1
    assert result.alignments["itermax"] == ((0, 0), (1, 1))
    assert result.similarities == ((1.0, 0.0), (0.0, 1.0))
    assert all(sum(row) == pytest.approx(1.0) for row in result.source_to_target_probabilities)


def test_conservative_repair_adds_a_nearby_mwmf_link_when_both_tokens_are_unaligned() -> None:
    itermax_links = {(0, 0), (1, 2), (2, 2), (4, 2), (5, 1), (6, 4)}
    backend = StubBackend(itermax_links, {"mwmf": itermax_links | {(3, 3)}})
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "ar",
            "method": "itermax",
            "repair": {
                "enabled": True,
                "strategy": "conservative",
                "max_position_distance": 0.35,
            },
            "sentence_pairs": [
                {
                    "source": "This is a machine translation example.",
                    "target": "هذه مثال لترجمة آلة.",
                }
            ],
        }
    )

    result = service.align(request).sentence_alignments[0]

    repaired_link = next(link for link in result.links if link.origin == "repaired")
    assert result.source_tokens[repaired_link.source_index].text == "machine"
    assert result.target_tokens[repaired_link.target_index].text == "آلة"
    assert repaired_link.similarity == 0.9
    assert repaired_link.confidence >= request.repair.min_confidence
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


@pytest.mark.parametrize("min_confidence", [0.35, 0.99])
def test_conservative_repair_collapses_ambiguous_low_coverage_clauses_into_phrase_groups(
    min_confidence: float,
) -> None:
    itermax_links = {
        (0, 0),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 9),
        (6, 12),
        (7, 13),
    }
    backend = StubBackend(itermax_links, {"mwmf": itermax_links | {(5, 7)}})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_direction_phrase_tokenize),
        backend_factory=lambda: backend,
    )
    request = AlignmentRequest.model_validate(
        {
            "source_language": "zh-Hans",
            "target_language": "en",
            "method": "itermax",
            "repair": {
                "enabled": True,
                "strategy": "conservative",
                "max_position_distance": 0.35,
                "min_similarity": 0.45,
                "min_confidence": min_confidence,
            },
            "sentence_pairs": [
                {
                    "id": "sentence-1",
                    "source": "北方的北方，南方的南方。",
                    "target": "The far north of the North, the far south of the South.",
                }
            ],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert [(link.source_index, link.target_index) for link in result.links] == [(3, 6), (7, 13)]
    refined_groups = [group for group in result.alignment_groups if group.origin == "refined"]
    assert [group.type for group in refined_groups] == ["many-to-many", "many-to-many"]
    assert [group.source_indices for group in refined_groups] == [[0, 1, 2], [4, 5, 6]]
    assert [group.target_indices for group in refined_groups] == [
        [0, 1, 2, 3, 4, 5],
        [7, 8, 9, 10, 11, 12],
    ]
    assert all(group.links == [] for group in refined_groups)
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


def test_conservative_repair_segments_a_long_clause_into_local_monotonic_spans() -> None:
    itermax_links = {(0, 0), (1, 1), (4, 2), (5, 3), (6, 4)}
    backend = StubBackend(
        itermax_links,
        {"mwmf": itermax_links},
        similarities=[
            [0.95, 0.80, 0.70, 0.60, 0.50],
            [0.90, 0.91, 0.70, 0.60, 0.50],
            [0.80, 0.90, 0.70, 0.60, 0.50],
            [0.70, 0.70, 0.92, 0.70, 0.50],
            [0.70, 0.70, 0.94, 0.70, 0.50],
            [0.70, 0.70, 0.70, 0.92, 0.50],
            [0.50, 0.50, 0.50, 0.50, 0.98],
        ],
        source_embeddings=[[1.0, 0.0]] * 7,
        target_embeddings=[[1.0, 0.0]] * 5,
    )
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_machine_translation_tokenize),
        backend_factory=lambda: backend,
    )
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "zh-Hans",
            "method": "itermax",
            "repair": {
                "enabled": True,
                "strategy": "conservative",
                "max_source_span": 3,
                "max_target_span": 6,
                "min_span_coverage": 0.75,
            },
            "sentence_pairs": [
                {
                    "id": "sentence-1",
                    "source": "This is a machine translation example.",
                    "target": "这是一个机器翻译示例。",
                }
            ],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert [(link.source_index, link.target_index) for link in result.links] == [(5, 3), (6, 4)]
    refined_groups = [group for group in result.alignment_groups if group.origin == "refined"]
    assert [group.type for group in refined_groups] == ["many-to-one", "one-to-one", "many-to-one"]
    assert [group.source_indices for group in refined_groups] == [[0, 1], [2], [3, 4]]
    assert [group.target_indices for group in refined_groups] == [[0], [1], [2]]
    assert all(group.links == [] for group in refined_groups)
    assert result.alignment_groups[-2].source_tokens == ["example"]
    assert result.alignment_groups[-2].target_tokens == ["示例"]
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


def test_local_monotonic_span_segmentation_is_symmetric() -> None:
    itermax_links = {(0, 0), (1, 1), (2, 4), (3, 5), (4, 6)}
    similarities = np.asarray(
        [
            [0.95, 0.80, 0.70, 0.60, 0.50],
            [0.90, 0.91, 0.70, 0.60, 0.50],
            [0.80, 0.90, 0.70, 0.60, 0.50],
            [0.70, 0.70, 0.92, 0.70, 0.50],
            [0.70, 0.70, 0.94, 0.70, 0.50],
            [0.70, 0.70, 0.70, 0.92, 0.50],
            [0.50, 0.50, 0.50, 0.50, 0.98],
        ]
    ).transpose()
    backend = StubBackend(
        itermax_links,
        {"mwmf": itermax_links},
        similarities=similarities.tolist(),
        source_embeddings=[[1.0, 0.0]] * 5,
        target_embeddings=[[1.0, 0.0]] * 7,
    )
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_machine_translation_tokenize),
        backend_factory=lambda: backend,
    )
    request = AlignmentRequest.model_validate(
        {
            "source_language": "zh-Hans",
            "target_language": "en",
            "repair": {"strategy": "conservative"},
            "sentence_pairs": [
                {
                    "source": "这是一个机器翻译示例。",
                    "target": "This is a machine translation example.",
                }
            ],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert [(link.source_index, link.target_index) for link in result.links] == [(3, 5), (4, 6)]
    refined_groups = [group for group in result.alignment_groups if group.origin == "refined"]
    assert [group.source_indices for group in refined_groups] == [[0], [1], [2]]
    assert [group.target_indices for group in refined_groups] == [[0, 1], [2], [3, 4]]
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


def test_conservative_repair_groups_a_local_crossing_and_rule_aligns_arrow_expressions() -> None:
    itermax_links = {(0, 3), (1, 2), (2, 0), (3, 1), (4, 6), (5, 5), (6, 4)}
    backend = StubBackend(itermax_links, {"mwmf": itermax_links})
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_atomic_link_tokenize),
        backend_factory=lambda: backend,
    )
    request = AlignmentRequest.model_validate(
        {
            "source_language": "zh-Hans",
            "target_language": "en",
            "repair": {"strategy": "conservative"},
            "sentence_pairs": [
                {
                    "source": "原子链接只保留 5→3 和 6→4",
                    "target": "Only retain atomic links 5→3 and 6→4",
                }
            ],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert [token.text for token in result.source_tokens] == [
        "原子",
        "链接",
        "只",
        "保留",
        "5→3",
        "和",
        "6→4",
    ]
    assert [(link.source_index, link.target_index, link.origin) for link in result.links] == [
        (2, 0, "model"),
        (3, 1, "model"),
        (4, 4, "rule"),
        (5, 5, "model"),
        (6, 6, "rule"),
    ]
    refined_group = next(group for group in result.alignment_groups if group.origin == "refined")
    assert refined_group.source_indices == [0, 1]
    assert refined_group.target_indices == [2, 3]
    assert refined_group.source_tokens == ["原子", "链接"]
    assert refined_group.target_tokens == ["atomic", "links"]
    assert refined_group.links == []
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


def test_span_aware_repair_expands_one_side_when_the_pooled_embedding_score_improves() -> None:
    links = {(0, 0), (1, 2)}
    backend = StubBackend(
        links,
        {"mwmf": links},
        similarities=[[0.85, 0.8, 0.1], [0.1, 0.1, 0.95]],
        source_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        target_embeddings=[[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]],
    )
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "en",
            "repair": {"strategy": "span-aware"},
            "sentence_pairs": [{"source": "NewYork.", "target": "New York."}],
        }
    )

    result = service.align(request).sentence_alignments[0]

    refined_group = next(group for group in result.alignment_groups if group.origin == "refined")
    assert refined_group.type == "one-to-many"
    assert refined_group.source_indices == [0]
    assert refined_group.target_indices == [0, 1]
    assert [(link.source_index, link.target_index) for link in refined_group.links] == [(0, 0)]
    assert refined_group.similarity == 1.0
    assert result.unaligned_source_indices == []
    assert result.unaligned_target_indices == []


def test_conservative_repair_respects_the_position_distance_limit() -> None:
    backend = StubBackend({(1, 1)}, {"mwmf": {(0, 2)}})
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "en",
            "repair": {"max_position_distance": 0.1},
            "sentence_pairs": [{"source": "zero one", "target": "zero one two"}],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert all(link.origin != "repaired" for link in result.links)
    assert result.unaligned_source_indices == [0]
    assert result.unaligned_target_indices == [0, 2]


def test_conservative_repair_does_not_attach_to_an_already_aligned_target() -> None:
    backend = StubBackend({(0, 0)}, {"mwmf": {(1, 0)}})
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "en",
            "repair": {},
            "sentence_pairs": [{"source": "hello extra", "target": "hello"}],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert [(link.source_index, link.target_index, link.origin) for link in result.links] == [(0, 0, "model")]
    assert result.unaligned_source_indices == [1]


def test_conservative_repair_rejects_a_low_similarity_candidate() -> None:
    backend = StubBackend(
        {(0, 0)},
        {"mwmf": {(0, 0), (1, 1)}},
        similarities=[[0.9, 0.1], [0.1, 0.3]],
    )
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "en",
            "repair": {"min_similarity": 0.45, "min_confidence": 0.0},
            "sentence_pairs": [{"source": "hello extra", "target": "hello omitted"}],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert all(link.origin != "repaired" for link in result.links)
    assert result.unaligned_source_indices == [1]
    assert result.unaligned_target_indices == [1]


def test_conservative_repair_rejects_a_candidate_below_the_confidence_threshold() -> None:
    backend = StubBackend(
        {(0, 0)},
        {"mwmf": {(0, 0), (1, 1)}},
        similarities=[[0.9, 0.1], [0.1, 0.9]],
    )
    service = AlignmentService(Settings(), backend_factory=lambda: backend)
    request = AlignmentRequest.model_validate(
        {
            "source_language": "en",
            "target_language": "en",
            "repair": {"min_similarity": 0.0, "min_confidence": 0.99},
            "sentence_pairs": [{"source": "hello extra", "target": "hello candidate"}],
        }
    )

    result = service.align(request).sentence_alignments[0]

    assert all(link.origin != "repaired" for link in result.links)
    assert result.unaligned_source_indices == [1]
    assert result.unaligned_target_indices == [1]


def test_confidence_is_higher_for_a_clear_mutual_match_than_an_ambiguous_match() -> None:
    backend = StubBackend(
        {(0, 0), (1, 1)},
        {"inter": {(0, 0)}, "mwmf": {(0, 0), (1, 1)}},
        similarities=[[0.95, 0.1], [0.7, 0.7]],
    )
    service = AlignmentService(
        Settings(),
        tokenizer=WordTokenizer(chinese_tokenize=_whitespace_tokenize),
        backend_factory=lambda: backend,
    )

    result = service.align(_request("clear ambiguous", "明确 模糊")).sentence_alignments[0]
    scores = {(link.source_index, link.target_index): link.confidence for link in result.links}

    assert scores[(0, 0)] > scores[(1, 1)]
