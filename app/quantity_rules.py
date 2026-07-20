from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.schemas import Token

RawLink = tuple[int, int]
QuantityKey = tuple[str, str]

_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_PERCENT_PATTERN = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+))%$")
_PERCENT_SIGNS = {"%", "％"}


@dataclass(frozen=True, slots=True)
class QuantitySpan:
    key: QuantityKey
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuantityRuleAlignment:
    links: frozenset[RawLink]
    source_indices: frozenset[int]
    target_indices: frozenset[int]


def align_quantity_spans(
    source_tokens: list[Token],
    target_tokens: list[Token],
) -> QuantityRuleAlignment:
    """Align equivalent percentages even when token boundaries differ."""

    source_by_key = _group_spans_by_key(_find_quantity_spans(source_tokens))
    target_by_key = _group_spans_by_key(_find_quantity_spans(target_tokens))
    links: set[RawLink] = set()
    matched_source: set[int] = set()
    matched_target: set[int] = set()

    for key in source_by_key.keys() & target_by_key.keys():
        for source_span, target_span in zip(
            source_by_key[key],
            target_by_key[key],
            strict=False,
        ):
            matched_source.update(source_span.indices)
            matched_target.update(target_span.indices)
            links.update(
                (source_index, target_index)
                for source_index in source_span.indices
                for target_index in target_span.indices
            )

    return QuantityRuleAlignment(
        links=frozenset(links),
        source_indices=frozenset(matched_source),
        target_indices=frozenset(matched_target),
    )


def _group_spans_by_key(
    spans: list[QuantitySpan],
) -> dict[QuantityKey, list[QuantitySpan]]:
    grouped: dict[QuantityKey, list[QuantitySpan]] = defaultdict(list)
    for span in spans:
        grouped[span.key].append(span)
    return grouped


def _find_quantity_spans(tokens: list[Token]) -> list[QuantitySpan]:
    spans: list[QuantitySpan] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.is_protected:
            index += 1
            continue

        text = unicodedata.normalize("NFKC", token.text.strip())
        percent = _PERCENT_PATTERN.fullmatch(text)
        if percent is not None:
            spans.append(
                QuantitySpan(
                    key=("percent", _normalize_number(percent.group(1))),
                    indices=(index,),
                )
            )
            index += 1
            continue

        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            _NUMBER_PATTERN.fullmatch(text)
            and next_token is not None
            and not next_token.is_protected
            and unicodedata.normalize("NFKC", next_token.text.strip()) in _PERCENT_SIGNS
        ):
            spans.append(
                QuantitySpan(
                    key=("percent", _normalize_number(text)),
                    indices=(index, index + 1),
                )
            )
            index += 2
            continue

        index += 1
    return spans


def _normalize_number(text: str) -> str:
    try:
        value = Decimal(text)
    except InvalidOperation:
        return text
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized
