from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.schemas import Token

RawLink = tuple[int, int]
TemporalKey = tuple[str, int]

_YEAR_PATTERN = re.compile(r"^[12]\d{3}$")
_YEAR_WITH_UNIT_PATTERN = re.compile(r"^([12]\d{3})年$")
_MONTH_WITH_UNIT_PATTERN = re.compile(r"^(\d{1,2})月$")
_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_CHINESE_MONTHS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


@dataclass(frozen=True, slots=True)
class TemporalSpan:
    key: TemporalKey
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalRuleAlignment:
    links: frozenset[RawLink]
    source_indices: frozenset[int]
    target_indices: frozenset[int]


def align_temporal_spans(
    source_tokens: list[Token],
    target_tokens: list[Token],
) -> TemporalRuleAlignment:
    source_by_key = _group_spans_by_key(_find_temporal_spans(source_tokens))
    target_by_key = _group_spans_by_key(_find_temporal_spans(target_tokens))
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

    return TemporalRuleAlignment(
        links=frozenset(links),
        source_indices=frozenset(matched_source),
        target_indices=frozenset(matched_target),
    )


def _group_spans_by_key(
    spans: list[TemporalSpan],
) -> dict[TemporalKey, list[TemporalSpan]]:
    grouped: dict[TemporalKey, list[TemporalSpan]] = defaultdict(list)
    for span in spans:
        grouped[span.key].append(span)
    return grouped


def _find_temporal_spans(tokens: list[Token]) -> list[TemporalSpan]:
    spans: list[TemporalSpan] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.is_protected:
            index += 1
            continue

        text = token.text.strip()
        normalized = text.casefold().rstrip(".")
        month = _MONTH_NAMES.get(normalized)
        if month is not None:
            spans.append(TemporalSpan(key=("month", month), indices=(index,)))
            index += 1
            continue

        year_with_unit = _YEAR_WITH_UNIT_PATTERN.fullmatch(text)
        if year_with_unit is not None:
            spans.append(
                TemporalSpan(
                    key=("year", int(year_with_unit.group(1))),
                    indices=(index,),
                )
            )
            index += 1
            continue

        month_with_unit = _MONTH_WITH_UNIT_PATTERN.fullmatch(text)
        if month_with_unit is not None:
            month_value = int(month_with_unit.group(1))
            if 1 <= month_value <= 12:
                spans.append(TemporalSpan(key=("month", month_value), indices=(index,)))
            index += 1
            continue

        chinese_month = _parse_chinese_month(text)
        if chinese_month is not None:
            spans.append(TemporalSpan(key=("month", chinese_month), indices=(index,)))
            index += 1
            continue

        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if next_token is not None and not next_token.is_protected:
            if _YEAR_PATTERN.fullmatch(text) and next_token.text == "年":
                spans.append(
                    TemporalSpan(
                        key=("year", int(text)),
                        indices=(index, index + 1),
                    )
                )
                index += 2
                continue

            month_value = _parse_month_number(text)
            if month_value is not None and next_token.text == "月":
                spans.append(
                    TemporalSpan(
                        key=("month", month_value),
                        indices=(index, index + 1),
                    )
                )
                index += 2
                continue

        if _YEAR_PATTERN.fullmatch(text):
            spans.append(TemporalSpan(key=("year", int(text)), indices=(index,)))
        index += 1
    return spans


def _parse_month_number(text: str) -> int | None:
    if text.isdigit():
        value = int(text)
        return value if 1 <= value <= 12 else None
    return _CHINESE_MONTHS.get(text)


def _parse_chinese_month(text: str) -> int | None:
    if not text.endswith("月"):
        return None
    return _CHINESE_MONTHS.get(text[:-1])
