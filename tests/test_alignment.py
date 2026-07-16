from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.alignment import AlignmentModelError, AlignmentService, SimAlignBackend, _ensure_device_available
from app.config import Settings
from app.schemas import AlignmentRequest
from app.tokenization import WordTokenizer


def _whitespace_tokenize(value: str):
    for match in re.finditer(r"\S+", value):
        yield match.group(), match.start(), match.end()


class StubBackend:
    def __init__(
        self,
        links: set[tuple[int, int]],
        additional_alignments: dict[str, set[tuple[int, int]]] | None = None,
    ) -> None:
        self._alignments = {"itermax": links, **(additional_alignments or {})}
        self.calls = 0

    def get_word_aligns(self, source_tokens: list[str], target_tokens: list[str]):
        self.calls += 1
        return self._alignments


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
