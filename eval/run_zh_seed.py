from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "eval" / "data" / "zh_en" / "capability_seed.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
CONFIG_NAMES = ("raw", "conservative", "span-aware")
KNOWN_FLAGS = {"no_whole_sentence_group"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class GoldPair:
    source: Span
    target: Span
    source_occurrence: int
    target_occurrence: int


@dataclass
class Example:
    index: int
    id: str
    category: str
    source: str
    target: str
    source_language: str
    target_language: str
    gold: tuple[GoldPair, ...]
    forbidden: tuple[GoldPair, ...]
    flags: frozenset[str]
    raw: dict[str, Any]


@dataclass
class PairResult:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    group_exact: bool
    group_contained: bool
    group_partial: bool
    link_hit: bool
    rule_link_hit: bool
    evidence_hit: bool
    source_coverage: float
    target_coverage: float
    token_coverage: float
    full_coverage: bool
    protected_pair: bool
    protected_pair_hit: bool
    hit_origins: tuple[str, ...]
    matching_groups: tuple[dict[str, Any], ...] = ()


@dataclass
class ExampleResult:
    id: str
    category: str
    config: str
    method: str
    source: str
    target: str
    gold_pairs: list[PairResult] = field(default_factory=list)
    forbidden_rule_hits: list[dict[str, Any]] = field(default_factory=list)
    forbidden_any_hits: list[dict[str, Any]] = field(default_factory=list)
    whole_sentence_group_violation: bool = False
    whole_sentence_groups: list[dict[str, Any]] = field(default_factory=list)
    rule_links: list[dict[str, Any]] = field(default_factory=list)
    unmatched_rule_links: list[dict[str, Any]] = field(default_factory=list)
    origin_counts: dict[str, int] = field(default_factory=dict)
    group_origin_counts: dict[str, int] = field(default_factory=dict)
    source_token_count: int = 0
    target_token_count: int = 0
    error: str | None = None


def _find_occurrences(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    cursor = 0
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return positions
        positions.append(position)
        cursor = position + 1


def _resolve_span(
    text: str,
    needle: str,
    occurrence: int | None,
    *,
    context: str,
    warnings: list[str],
) -> tuple[Span, int]:
    if not needle:
        raise ValueError(f"{context}: empty span")
    positions = _find_occurrences(text, needle)
    if not positions:
        raise ValueError(f"{context}: span {needle!r} not found in {text!r}")
    if occurrence is None:
        if len(positions) > 1:
            warnings.append(
                f"{context}: span {needle!r} occurs {len(positions)} times; defaulting to occurrence 0"
            )
        occurrence = 0
    if not 0 <= occurrence < len(positions):
        raise ValueError(
            f"{context}: occurrence {occurrence} out of range for {needle!r} "
            f"({len(positions)} occurrence(s))"
        )
    start = positions[occurrence]
    return Span(start=start, end=start + len(needle), text=needle), occurrence


def _parse_pair(
    raw: dict[str, Any],
    *,
    source: str,
    target: str,
    context: str,
    warnings: list[str],
) -> GoldPair:
    source_span, source_occurrence = _resolve_span(
        source,
        str(raw["source"]),
        raw.get("source_occurrence"),
        context=f"{context}.source",
        warnings=warnings,
    )
    target_span, target_occurrence = _resolve_span(
        target,
        str(raw["target"]),
        raw.get("target_occurrence"),
        context=f"{context}.target",
        warnings=warnings,
    )
    return GoldPair(
        source=source_span,
        target=target_span,
        source_occurrence=source_occurrence,
        target_occurrence=target_occurrence,
    )


def load_examples(path: Path) -> tuple[list[Example], list[str]]:
    warnings: list[str] = []
    examples: list[Example] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        example_id = str(raw["id"])
        if example_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {example_id!r}")
        seen_ids.add(example_id)

        flags = frozenset(raw.get("flags", []))
        unknown_flags = flags - KNOWN_FLAGS
        if unknown_flags:
            raise ValueError(f"{path}:{line_number}: unknown flags {sorted(unknown_flags)}")

        source = str(raw["source"])
        target = str(raw["target"])
        gold = tuple(
            _parse_pair(
                item,
                source=source,
                target=target,
                context=f"{example_id}.gold[{index}]",
                warnings=warnings,
            )
            for index, item in enumerate(raw.get("gold", []))
        )
        if not gold:
            raise ValueError(f"{path}:{line_number}: at least one gold pair is required")
        forbidden = tuple(
            _parse_pair(
                item,
                source=source,
                target=target,
                context=f"{example_id}.forbidden[{index}]",
                warnings=warnings,
            )
            for index, item in enumerate(raw.get("forbidden_rules", []))
        )
        examples.append(
            Example(
                index=len(examples),
                id=example_id,
                category=str(raw.get("category", "uncategorized")),
                source=source,
                target=target,
                source_language=str(raw.get("source_language", "en")),
                target_language=str(raw.get("target_language", "zh-Hans")),
                gold=gold,
                forbidden=forbidden,
                flags=flags,
                raw=raw,
            )
        )
    return examples, warnings


def _repair_payload(config: str) -> dict[str, Any] | None:
    if config == "raw":
        return None
    return {"enabled": True, "strategy": config}


def _request_payload(
    examples: Sequence[Example],
    *,
    source_language: str,
    target_language: str,
    method: str,
    config: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_language": source_language,
        "target_language": target_language,
        "method": method,
        "sentence_pairs": [
            {"id": example.id, "source": example.source, "target": example.target}
            for example in examples
        ],
    }
    repair = _repair_payload(config)
    if repair is not None:
        payload["repair"] = repair
    return payload


def _align_batch(
    client: httpx.Client,
    api_url: str,
    examples: Sequence[Example],
    *,
    source_language: str,
    target_language: str,
    method: str,
    config: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _request_payload(
        examples,
        source_language=source_language,
        target_language=target_language,
        method=method,
        config=config,
    )
    response = client.post(api_url, json=payload)
    response.raise_for_status()
    body = response.json()
    alignments = body.get("sentence_alignments", [])
    if len(alignments) != len(examples):
        raise RuntimeError(f"Expected {len(examples)} alignments, received {len(alignments)}")
    return {str(item["id"]): item for item in alignments}, {
        key: body.get(key)
        for key in ("model", "embedding_layer", "method", "confidence_method")
    }


def fetch_alignments(
    client: httpx.Client,
    api_url: str,
    examples: Sequence[Example],
    *,
    method: str,
    config: str,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        grouped[(example.source_language, example.target_language)].append(example)

    for (source_language, target_language), group in grouped.items():
        for offset in range(0, len(group), batch_size):
            batch = group[offset : offset + batch_size]
            try:
                batch_results, batch_metadata = _align_batch(
                    client,
                    api_url,
                    batch,
                    source_language=source_language,
                    target_language=target_language,
                    method=method,
                    config=config,
                )
                results.update(batch_results)
                metadata.update({key: value for key, value in batch_metadata.items() if value is not None})
            except Exception:
                for example in batch:
                    try:
                        single, single_metadata = _align_batch(
                            client,
                            api_url,
                            [example],
                            source_language=source_language,
                            target_language=target_language,
                            method=method,
                            config=config,
                        )
                        results.update(single)
                        metadata.update(
                            {key: value for key, value in single_metadata.items() if value is not None}
                        )
                    except Exception as exc:  # noqa: BLE001 - keep evaluating remaining examples
                        failures[example.id] = str(exc)
    return results, failures, metadata


def _token_indices(tokens: Sequence[dict[str, Any]], span: Span) -> tuple[int, ...]:
    contained = tuple(
        int(token["index"])
        for token in tokens
        if span.start <= int(token["start"]) and int(token["end"]) <= span.end
    )
    if contained:
        return contained

    best_overlap = 0
    best_indices: list[int] = []
    for token in tokens:
        overlap = min(span.end, int(token["end"])) - max(span.start, int(token["start"]))
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_indices = [int(token["index"])]
        elif overlap == best_overlap:
            best_indices.append(int(token["index"]))
    return tuple(best_indices)


def _is_content_token(token: dict[str, Any]) -> bool:
    return any(
        unicodedata.category(character)[0] not in {"P", "Z", "C"}
        for character in str(token["text"])
    )


def _coverage(
    *,
    tokens_by_side: dict[str, Sequence[dict[str, Any]]],
    source_indices: set[int],
    target_indices: set[int],
    groups: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
) -> tuple[float, float, float, bool, bool]:
    covered_source: set[int] = set()
    covered_target: set[int] = set()
    for link in links:
        source_index = int(link["source_index"])
        target_index = int(link["target_index"])
        if source_index in source_indices and target_index in target_indices:
            covered_source.add(source_index)
            covered_target.add(target_index)
    for group in groups:
        group_source = {int(index) for index in group["source_indices"]} & source_indices
        group_target = {int(index) for index in group["target_indices"]} & target_indices
        if group_source and group_target:
            covered_source.update(group_source)
            covered_target.update(group_target)

    content_source = {
        index
        for index in source_indices
        if _is_content_token(tokens_by_side["source"][index])
    }
    content_target = {
        index
        for index in target_indices
        if _is_content_token(tokens_by_side["target"][index])
    }
    source_coverage = (
        len(content_source & covered_source) / len(content_source) if content_source else 1.0
    )
    target_coverage = (
        len(content_target & covered_target) / len(content_target) if content_target else 1.0
    )
    total = len(content_source) + len(content_target)
    token_coverage = (
        (len(content_source & covered_source) + len(content_target & covered_target)) / total
        if total
        else 1.0
    )
    evidence_hit = bool(covered_source or covered_target)
    return (
        round(source_coverage, 6),
        round(target_coverage, 6),
        round(token_coverage, 6),
        evidence_hit,
        source_coverage == 1.0 and target_coverage == 1.0,
    )


def _dedupe_groups(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    for group in groups:
        key = (
            tuple(sorted(int(index) for index in group["source_indices"])),
            tuple(sorted(int(index) for index in group["target_indices"])),
        )
        unique.setdefault(key, group)
    return list(unique.values())


def _group_matches_pair(group: dict[str, Any], source: set[int], target: set[int]) -> str:
    group_source = {int(index) for index in group["source_indices"]}
    group_target = {int(index) for index in group["target_indices"]}
    if group_source == source and group_target == target:
        return "exact"
    if source <= group_source and target <= group_target:
        return "contained"
    if group_source & source and group_target & target:
        return "partial"
    return "none"


def _hit_origins(
    links: Sequence[dict[str, Any]], source: set[int], target: set[int]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(link["origin"])
                for link in links
                if int(link["source_index"]) in source and int(link["target_index"]) in target
            }
        )
    )


def _pair_result(
    gold: GoldPair,
    *,
    tokens_by_side: dict[str, Sequence[dict[str, Any]]],
    groups: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
) -> PairResult:
    source_indices = _token_indices(tokens_by_side["source"], gold.source)
    target_indices = _token_indices(tokens_by_side["target"], gold.target)
    if not source_indices or not target_indices:
        missing = "source" if not source_indices else "target"
        raise ValueError(f"gold pair {gold.source.text!r}->{gold.target.text!r} maps to no {missing} token")
    source_set = set(source_indices)
    target_set = set(target_indices)
    matches = [
        (group, _group_matches_pair(group, source_set, target_set))
        for group in groups
        if _group_matches_pair(group, source_set, target_set) != "none"
    ]
    group_exact = any(match == "exact" for _, match in matches)
    group_contained = any(match in {"exact", "contained"} for _, match in matches)
    group_partial = any(match == "partial" for _, match in matches)
    origins = _hit_origins(links, source_set, target_set)
    source_coverage, target_coverage, token_coverage, evidence_hit, full_coverage = _coverage(
        tokens_by_side=tokens_by_side,
        source_indices=source_set,
        target_indices=target_set,
        groups=groups,
        links=links,
    )
    protected_pair = all(
        bool(tokens_by_side["source"][index].get("is_protected")) for index in source_indices
    ) and all(
        bool(tokens_by_side["target"][index].get("is_protected")) for index in target_indices
    )
    return PairResult(
        source_indices=source_indices,
        target_indices=target_indices,
        source_tokens=tuple(
            str(tokens_by_side["source"][index]["text"]) for index in source_indices
        ),
        target_tokens=tuple(
            str(tokens_by_side["target"][index]["text"]) for index in target_indices
        ),
        group_exact=group_exact,
        group_contained=group_contained,
        group_partial=group_partial,
        link_hit=bool(origins),
        rule_link_hit="rule" in origins,
        evidence_hit=evidence_hit,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        token_coverage=token_coverage,
        full_coverage=full_coverage,
        protected_pair=protected_pair,
        protected_pair_hit=protected_pair and (group_exact or bool(origins)),
        hit_origins=origins,
        matching_groups=tuple(
            {
                "type": group["type"],
                "origin": group["origin"],
                "source_tokens": group["source_tokens"],
                "target_tokens": group["target_tokens"],
                "match": match,
            }
            for group, match in matches
        ),
    )


def _link_record(link: dict[str, Any], tokens_by_side: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    source_index = int(link["source_index"])
    target_index = int(link["target_index"])
    return {
        "source_index": source_index,
        "target_index": target_index,
        "source_token": tokens_by_side["source"][source_index]["text"],
        "target_token": tokens_by_side["target"][target_index]["text"],
        "origin": link["origin"],
        "similarity": link["similarity"],
        "confidence": link["confidence"],
    }


def score_example(
    example: Example, response: dict[str, Any], *, config: str, method: str
) -> ExampleResult:
    source_tokens = list(response["source_tokens"])
    target_tokens = list(response["target_tokens"])
    tokens_by_side = {"source": source_tokens, "target": target_tokens}
    groups = _dedupe_groups(response.get("alignment_groups", []))
    links = list(response.get("links", []))
    origin_counts = {
        origin: sum(1 for link in links if link["origin"] == origin)
        for origin in sorted({link["origin"] for link in links})
    }
    group_origin_counts = {
        origin: sum(1 for group in groups if group["origin"] == origin)
        for origin in sorted({group["origin"] for group in groups})
    }
    result = ExampleResult(
        id=example.id,
        category=example.category,
        config=config,
        method=method,
        source=example.source,
        target=example.target,
        source_token_count=len(source_tokens),
        target_token_count=len(target_tokens),
        origin_counts=origin_counts,
        group_origin_counts=group_origin_counts,
    )

    for gold in example.gold:
        result.gold_pairs.append(
            _pair_result(
                gold,
                tokens_by_side=tokens_by_side,
                groups=groups,
                links=links,
            )
        )

    for forbidden in example.forbidden:
        source_indices = set(_token_indices(source_tokens, forbidden.source))
        target_indices = set(_token_indices(target_tokens, forbidden.target))
        if not source_indices or not target_indices:
            continue
        rule_pairs: dict[tuple[int, int], dict[str, Any]] = {}
        any_pairs: dict[tuple[int, int], dict[str, Any]] = {}
        for link in links:
            source_index = int(link["source_index"])
            target_index = int(link["target_index"])
            if source_index not in source_indices or target_index not in target_indices:
                continue
            record = _link_record(link, tokens_by_side)
            record["forbidden_source"] = forbidden.source.text
            record["forbidden_target"] = forbidden.target.text
            any_pairs[(source_index, target_index)] = record
            if link["origin"] == "rule":
                rule_pairs[(source_index, target_index)] = record
        for group in groups:
            group_source = {int(index) for index in group["source_indices"]} & source_indices
            group_target = {int(index) for index in group["target_indices"]} & target_indices
            if not group_source or not group_target:
                continue
            for source_index in group_source:
                for target_index in group_target:
                    if (source_index, target_index) in any_pairs:
                        continue
                    any_pairs[(source_index, target_index)] = {
                        "source_index": source_index,
                        "target_index": target_index,
                        "source_token": source_tokens[source_index]["text"],
                        "target_token": target_tokens[target_index]["text"],
                        "origin": f"group:{group['origin']}",
                        "similarity": group["similarity"],
                        "confidence": group["confidence"],
                        "forbidden_source": forbidden.source.text,
                        "forbidden_target": forbidden.target.text,
                    }
        result.forbidden_rule_hits.extend(rule_pairs.values())
        result.forbidden_any_hits.extend(any_pairs.values())

    if "no_whole_sentence_group" in example.flags:
        for group in groups:
            source_coverage = len(group["source_indices"]) / max(1, len(source_tokens))
            target_coverage = len(group["target_indices"]) / max(1, len(target_tokens))
            if source_coverage >= 0.8 and target_coverage >= 0.8:
                result.whole_sentence_group_violation = True
                result.whole_sentence_groups.append(
                    {
                        "type": group["type"],
                        "origin": group["origin"],
                        "source_tokens": group["source_tokens"],
                        "target_tokens": group["target_tokens"],
                        "source_coverage": round(source_coverage, 4),
                        "target_coverage": round(target_coverage, 4),
                    }
                )

    for link in links:
        if link["origin"] != "rule":
            continue
        record = _link_record(link, tokens_by_side)
        result.rule_links.append(record)
        matched = any(
            int(link["source_index"]) in set(pair.source_indices)
            and int(link["target_index"]) in set(pair.target_indices)
            for pair in result.gold_pairs
        )
        if not matched:
            result.unmatched_rule_links.append(record)

    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _aggregate(results: Sequence[ExampleResult]) -> dict[str, Any]:
    pairs = [pair for result in results for pair in result.gold_pairs]
    gold_count = len(pairs)
    exact = sum(1 for pair in pairs if pair.group_exact)
    contained = sum(1 for pair in pairs if pair.group_contained)
    partial = sum(1 for pair in pairs if pair.group_partial)
    link_hit = sum(1 for pair in pairs if pair.link_hit)
    rule_hit = sum(1 for pair in pairs if pair.rule_link_hit)
    evidence_hit = sum(1 for pair in pairs if pair.evidence_hit)
    full_coverage = sum(1 for pair in pairs if pair.full_coverage)
    protected_pairs = sum(1 for pair in pairs if pair.protected_pair)
    protected_pair_hits = sum(1 for pair in pairs if pair.protected_pair_hit)
    mean_token_coverage = (
        round(sum(pair.token_coverage for pair in pairs) / len(pairs), 6) if pairs else None
    )
    rule_links = [link for result in results for link in result.rule_links]
    matched_rule_links = sum(
        len(result.rule_links) - len(result.unmatched_rule_links) for result in results
    )
    forbidden_rule_hits = [hit for result in results for hit in result.forbidden_rule_hits]
    forbidden_any_hits = [hit for result in results for hit in result.forbidden_any_hits]
    violations = [result.id for result in results if result.whole_sentence_group_violation]
    sentence_exact = sum(
        1 for result in results if result.gold_pairs and all(pair.group_exact for pair in result.gold_pairs)
    )
    sentence_contained = sum(
        1
        for result in results
        if result.gold_pairs and all(pair.group_contained for pair in result.gold_pairs)
    )
    sentence_link = sum(
        1 for result in results if result.gold_pairs and all(pair.link_hit for pair in result.gold_pairs)
    )
    sentence_evidence = sum(
        1 for result in results if result.gold_pairs and all(pair.evidence_hit for pair in result.gold_pairs)
    )
    sentence_full = sum(
        1 for result in results if result.gold_pairs and all(pair.full_coverage for pair in result.gold_pairs)
    )

    origin_counts: dict[str, int] = defaultdict(int)
    group_origin_counts: dict[str, int] = defaultdict(int)
    for result in results:
        for origin, count in result.origin_counts.items():
            origin_counts[origin] += count
        for origin, count in result.group_origin_counts.items():
            group_origin_counts[origin] += count

    return {
        "examples": len(results),
        "sentences_all_gold_exact": sentence_exact,
        "sentences_all_gold_contained": sentence_contained,
        "sentences_all_gold_linked": sentence_link,
        "sentences_all_gold_evidence": sentence_evidence,
        "sentences_all_gold_full_coverage": sentence_full,
        "gold_pairs": gold_count,
        "group_exact": exact,
        "group_contained": contained,
        "group_partial": partial,
        "link_hit": link_hit,
        "rule_link_hit": rule_hit,
        "evidence_hit": evidence_hit,
        "full_coverage": full_coverage,
        "group_exact_rate": _ratio(exact, gold_count),
        "group_contained_rate": _ratio(contained, gold_count),
        "group_partial_rate": _ratio(partial, gold_count),
        "link_hit_rate": _ratio(link_hit, gold_count),
        "rule_link_hit_rate": _ratio(rule_hit, gold_count),
        "evidence_hit_rate": _ratio(evidence_hit, gold_count),
        "full_coverage_rate": _ratio(full_coverage, gold_count),
        "mean_token_coverage": mean_token_coverage,
        "protected_pairs": protected_pairs,
        "protected_pair_hits": protected_pair_hits,
        "protected_pair_hit_rate": _ratio(protected_pair_hits, protected_pairs),
        "rule_links": len(rule_links),
        "matched_rule_links": matched_rule_links,
        "rule_link_precision": _ratio(matched_rule_links, len(rule_links)),
        "forbidden_rule_hits": len(forbidden_rule_hits),
        "forbidden_any_origin_hits": len(forbidden_any_hits),
        "whole_sentence_group_violations": violations,
        "link_origin_counts": dict(sorted(origin_counts.items())),
        "group_origin_counts": dict(sorted(group_origin_counts.items())),
    }


def _by_category(results: Sequence[ExampleResult]) -> dict[str, Any]:
    grouped: dict[str, list[ExampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)
    return {category: _aggregate(items) for category, items in sorted(grouped.items())}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _result_row(result: ExampleResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "category": result.category,
        "config": result.config,
        "method": result.method,
        "source": result.source,
        "target": result.target,
        "source_token_count": result.source_token_count,
        "target_token_count": result.target_token_count,
        "gold_pairs": [
            {
                "source_tokens": pair.source_tokens,
                "target_tokens": pair.target_tokens,
                "group_exact": pair.group_exact,
                "group_contained": pair.group_contained,
                "group_partial": pair.group_partial,
                "link_hit": pair.link_hit,
                "rule_link_hit": pair.rule_link_hit,
                "evidence_hit": pair.evidence_hit,
                "source_coverage": pair.source_coverage,
                "target_coverage": pair.target_coverage,
                "token_coverage": pair.token_coverage,
                "full_coverage": pair.full_coverage,
                "protected_pair": pair.protected_pair,
                "protected_pair_hit": pair.protected_pair_hit,
                "hit_origins": pair.hit_origins,
                "matching_groups": pair.matching_groups,
            }
            for pair in result.gold_pairs
        ],
        "forbidden_rule_hits": result.forbidden_rule_hits,
        "forbidden_any_origin_hits": result.forbidden_any_hits,
        "whole_sentence_group_violation": result.whole_sentence_group_violation,
        "whole_sentence_groups": result.whole_sentence_groups,
        "origin_counts": result.origin_counts,
        "group_origin_counts": result.group_origin_counts,
        "unmatched_rule_links": result.unmatched_rule_links,
        "error": result.error,
    }


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the live word-alignment API on the self-annotated EN-ZH capability seed set."
    )
    parser.add_argument("--api", default="http://localhost:8000/api/v1/align")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--method", default="itermax", choices=("itermax", "inter", "mwmf"))
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_NAMES), choices=CONFIG_NAMES)
    parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "zh_seed_summary.json")
    parser.add_argument("--sentences", type=Path, default=RESULTS_DIR / "zh_seed_sentences.jsonl")
    args = parser.parse_args(argv)

    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")

    examples, warnings = load_examples(args.data)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    categories: dict[str, int] = defaultdict(int)
    for example in examples:
        categories[example.category] += 1
    print(
        f"Validated {len(examples)} examples: "
        + ", ".join(f"{category}={count}" for category, count in sorted(categories.items())),
        file=sys.stderr,
    )
    if args.validate_only:
        return 0

    started_at = datetime.now(timezone.utc)
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    summary: dict[str, Any] = {
        "dataset": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
        "dataset_sha256": dataset_hash,
        "api": args.api,
        "method": args.method,
        "started_at": started_at.isoformat(),
        "examples": len(examples),
        "category_counts": dict(sorted(categories.items())),
        "warnings": warnings,
        "configs": {},
        "failures": {},
    }

    all_rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for config in args.configs:
            run_started = time.perf_counter()
            alignments, failures, metadata = fetch_alignments(
                client,
                args.api,
                examples,
                method=args.method,
                config=config,
                batch_size=args.batch_size,
            )
            summary["failures"].update({f"{config}:{key}": value for key, value in failures.items()})
            results: list[ExampleResult] = []
            for example in examples:
                response = alignments.get(example.id)
                if response is None:
                    continue
                try:
                    results.append(score_example(example, response, config=config, method=args.method))
                except Exception as exc:  # noqa: BLE001 - surface scoring errors instead of aborting
                    failed = ExampleResult(
                        id=example.id,
                        category=example.category,
                        config=config,
                        method=args.method,
                        source=example.source,
                        target=example.target,
                        error=str(exc),
                    )
                    results.append(failed)
            elapsed = time.perf_counter() - run_started
            aggregate = _aggregate(results)
            aggregate["elapsed_seconds"] = round(elapsed, 4)
            aggregate["failures"] = len(failures)
            summary["configs"][config] = {
                "metadata": metadata,
                "overall": aggregate,
                "by_category": _by_category(results),
            }
            all_rows.extend(_result_row(result) for result in results)

            print(
                f"{config}: {aggregate['examples']} examples, "
                f"exact={_format_rate(aggregate['group_exact_rate'])}, "
                f"contained={_format_rate(aggregate['group_contained_rate'])}, "
                f"link={_format_rate(aggregate['link_hit_rate'])}, "
                f"full={_format_rate(aggregate['full_coverage_rate'])}, "
                f"forbidden={aggregate['forbidden_any_origin_hits']}, "
                f"{elapsed:.2f}s",
                file=sys.stderr,
            )
            for category, metrics in summary["configs"][config]["by_category"].items():
                print(
                    f"  {category:<12} exact={_format_rate(metrics['group_exact_rate'])}, "
                    f"contained={_format_rate(metrics['group_contained_rate'])}, "
                    f"link={_format_rate(metrics['link_hit_rate'])}, "
                    f"full={_format_rate(metrics['full_coverage_rate'])}, "
                    f"forbidden={metrics['forbidden_any_origin_hits']}",
                    file=sys.stderr,
                )

    _write_json(args.output, summary)
    _write_jsonl(args.sentences, all_rows)
    print(f"Wrote {args.output} and {args.sentences}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
