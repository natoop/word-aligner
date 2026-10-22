from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
XLWA_DATA = ROOT / "eval" / "data" / "XL-WA" / "data"
RESULTS_DIR = ROOT / "eval" / "results"


@dataclass(frozen=True)
class Example:
    pair: str
    index: int
    source_text: str
    target_text: str
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    source_offsets: tuple[tuple[int, int], ...]
    target_offsets: tuple[tuple[int, int], ...]
    gold_links: frozenset[tuple[int, int]]
    possible_links: frozenset[tuple[int, int]]


@dataclass
class SentenceScore:
    pair: str
    config: str
    method: str
    index: int
    source: str
    target: str
    gold_count: int
    predicted_count: int
    tp: int
    fp: int
    fn: int
    strict_tp: int
    strict_fp: int
    strict_fn: int
    group_count: int
    supported_group_count: int
    gold_links_covered_by_groups: int
    overmerged_group_count: int
    gold_components: int
    fragmented_gold_components: int
    group_token_tp: int
    group_token_in_groups: int
    group_token_total: int
    unaligned_source_tp: int
    unaligned_source_fp: int
    unaligned_source_fn: int
    unaligned_target_tp: int
    unaligned_target_fp: int
    unaligned_target_fn: int
    confidences: tuple[float, ...]
    confidence_correct: tuple[bool, ...]
    similarities: tuple[float, ...]
    origins: tuple[str, ...]
    group_types: tuple[str, ...]
    error: str | None = None


CONFIG_NAMES = ("raw", "conservative", "span-aware")


def _repair_payload(config: str) -> dict[str, Any] | None:
    if config == "raw":
        return None
    return {"enabled": True, "strategy": config}


def _read_examples(pair: str, split: str, limit: int = 0) -> list[Example]:
    path = XLWA_DATA / pair / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XL-WA split: {path}")

    examples: list[Example] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected 3 tab-separated fields")
        source_tokens = tuple(fields[0].split())
        target_tokens = tuple(fields[1].split())
        if not source_tokens or not target_tokens:
            continue
        source_text, source_offsets = _join_tokens(source_tokens)
        target_text, target_offsets = _join_tokens(target_tokens)
        gold, possible = _parse_alignments(fields[2])
        _validate_links(gold, len(source_tokens), len(target_tokens), path, line_number)
        _validate_links(possible, len(source_tokens), len(target_tokens), path, line_number)
        examples.append(
            Example(
                pair=pair,
                index=len(examples),
                source_text=source_text,
                target_text=target_text,
                source_tokens=source_tokens,
                target_tokens=target_tokens,
                source_offsets=source_offsets,
                target_offsets=target_offsets,
                gold_links=frozenset(gold),
                possible_links=frozenset(possible),
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples


def _join_tokens(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for position, token in enumerate(tokens):
        if position:
            cursor += 1
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)
    return " ".join(tokens), tuple(offsets)


def _parse_alignments(raw: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    sure: set[tuple[int, int]] = set()
    possible: set[tuple[int, int]] = set()
    for item in raw.split():
        if "-" in item:
            source, target = item.split("-", maxsplit=1)
            sure.add((int(source), int(target)))
        elif "p" in item:
            source, target = item.split("p", maxsplit=1)
            possible.add((int(source), int(target)))
    return sure, possible


def _validate_links(
    links: Iterable[tuple[int, int]],
    source_count: int,
    target_count: int,
    path: Path,
    line_number: int,
) -> None:
    for source, target in links:
        if not (0 <= source < source_count and 0 <= target < target_count):
            raise ValueError(
                f"{path}:{line_number}: link {(source, target)} outside "
                f"{source_count}x{target_count} token matrix"
            )


def _overlap_indices(start: int, end: int, offsets: Sequence[tuple[int, int]]) -> list[int]:
    return [index for index, (gold_start, gold_end) in enumerate(offsets) if start < gold_end and end > gold_start]


def _best_index(start: int, end: int, offsets: Sequence[tuple[int, int]]) -> int:
    best_index = -1
    best_overlap = 0
    for index, (gold_start, gold_end) in enumerate(offsets):
        overlap = min(end, gold_end) - max(start, gold_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return best_index


def _gold_components(gold_links: Iterable[tuple[int, int]]) -> tuple[dict[tuple[str, int], int], int]:
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, target in gold_links:
        union(("s", source), ("t", target))

    roots: dict[tuple[str, int], int] = {}
    for node in list(parent):
        root = find(node)
        roots.setdefault(root, len(roots))
    component_by_node = {node: roots[find(node)] for node in parent}
    return component_by_node, len(roots)


def _group_gold_coverage(
    groups: Sequence[dict[str, Any]],
    api_source_tokens: Sequence[dict[str, Any]],
    api_target_tokens: Sequence[dict[str, Any]],
    source_offsets: Sequence[tuple[int, int]],
    target_offsets: Sequence[tuple[int, int]],
    gold: set[tuple[int, int]],
    component_by_node: dict[tuple[str, int], int],
) -> tuple[int, int, int, int, int, int]:
    covered: set[tuple[int, int]] = set()
    supported_groups = 0
    overmerged_groups = 0
    component_group_counts: dict[int, set[int]] = defaultdict(set)
    gold_aligned_source = {source for source, _ in gold}
    gold_aligned_target = {target for _, target in gold}
    token_in_groups: set[tuple[str, int]] = set()
    token_correct: set[tuple[str, int]] = set()

    for group_index, group in enumerate(groups):
        source_map: set[int] = set()
        target_map: set[int] = set()
        for source_index in group["source_indices"]:
            token = api_source_tokens[source_index]
            source_map.update(_overlap_indices(token["start"], token["end"], source_offsets))
        for target_index in group["target_indices"]:
            token = api_target_tokens[target_index]
            target_map.update(_overlap_indices(token["start"], token["end"], target_offsets))

        group_gold = {(source, target) for source in source_map for target in target_map} & gold
        if group_gold:
            supported_groups += 1
            covered.update(group_gold)
            for source, _ in group_gold:
                component_group_counts[component_by_node[("s", source)]].add(group_index)

        grouped_source = source_map & gold_aligned_source
        grouped_target = target_map & gold_aligned_target
        token_in_groups.update(("s", source) for source in grouped_source)
        token_in_groups.update(("t", target) for target in grouped_target)
        for source in grouped_source:
            if any((source, target) in gold for target in grouped_target):
                token_correct.add(("s", source))
        for target in grouped_target:
            if any((source, target) in gold for source in grouped_source):
                token_correct.add(("t", target))

        component_roots = {
            component_by_node[("s", source)] for source in source_map if ("s", source) in component_by_node
        }
        component_roots.update(
            component_by_node[("t", target)] for target in target_map if ("t", target) in component_by_node
        )
        if len(component_roots) > 1:
            overmerged_groups += 1

    fragmented_components = sum(1 for group_indices in component_group_counts.values() if len(group_indices) > 1)
    return (
        len(covered),
        supported_groups,
        overmerged_groups,
        fragmented_components,
        len(token_correct),
        len(token_in_groups),
    )


def _unaligned_scores(
    unaligned_indices: Sequence[int],
    api_tokens: Sequence[dict[str, Any]],
    gold_offsets: Sequence[tuple[int, int]],
    gold_aligned: set[int],
    all_count: int,
) -> tuple[int, int, int]:
    predicted_unaligned: set[int] = set()
    for index in unaligned_indices:
        token = api_tokens[index]
        predicted_unaligned.update(_overlap_indices(token["start"], token["end"], gold_offsets))
    gold_unaligned = set(range(all_count)) - gold_aligned
    tp = len(predicted_unaligned & gold_unaligned)
    fp = len(predicted_unaligned - gold_unaligned)
    fn = len(gold_unaligned - predicted_unaligned)
    return tp, fp, fn


def _score_sentence(
    example: Example,
    response: dict[str, Any],
    *,
    config: str,
    method: str,
) -> SentenceScore:
    source_offsets = example.source_offsets
    target_offsets = example.target_offsets
    gold = set(example.gold_links)

    projected: set[tuple[int, int]] = set()
    strict_projected: set[tuple[int, int]] = set()
    confidences: list[float] = []
    confidence_correct: list[bool] = []
    similarities: list[float] = []
    origins: list[str] = []

    for link in response["links"]:
        source_token = response["source_tokens"][link["source_index"]]
        target_token = response["target_tokens"][link["target_index"]]
        source_map = _overlap_indices(source_token["start"], source_token["end"], source_offsets)
        target_map = _overlap_indices(target_token["start"], target_token["end"], target_offsets)
        for source_index in source_map:
            for target_index in target_map:
                projected.add((source_index, target_index))

        source_best = _best_index(source_token["start"], source_token["end"], source_offsets)
        target_best = _best_index(target_token["start"], target_token["end"], target_offsets)
        strict_link = (source_best, target_best)
        if source_best >= 0 and target_best >= 0:
            strict_projected.add(strict_link)

        confidences.append(float(link["confidence"]))
        confidence_correct.append(strict_link in gold)
        similarities.append(float(link["similarity"]))
        origins.append(str(link.get("origin", "model")))

    tp = len(projected & gold)
    fp = len(projected - gold)
    fn = len(gold - projected)
    strict_tp = len(strict_projected & gold)
    strict_fp = len(strict_projected - gold)
    strict_fn = len(gold - strict_projected)

    groups = response["alignment_groups"]
    component_by_node, component_count = _gold_components(gold)
    (
        covered_by_groups,
        supported_groups,
        overmerged_groups,
        fragmented_components,
        group_token_tp,
        group_token_in_groups,
    ) = _group_gold_coverage(
        groups,
        response["source_tokens"],
        response["target_tokens"],
        source_offsets,
        target_offsets,
        gold,
        component_by_node,
    )

    gold_aligned_source = {source for source, _ in gold}
    gold_aligned_target = {target for _, target in gold}
    source_unaligned = _unaligned_scores(
        response["unaligned_source_indices"],
        response["source_tokens"],
        source_offsets,
        gold_aligned_source,
        len(example.source_tokens),
    )
    target_unaligned = _unaligned_scores(
        response["unaligned_target_indices"],
        response["target_tokens"],
        target_offsets,
        gold_aligned_target,
        len(example.target_tokens),
    )

    return SentenceScore(
        pair=example.pair,
        config=config,
        method=method,
        index=example.index,
        source=example.source_text,
        target=example.target_text,
        gold_count=len(gold),
        predicted_count=len(projected),
        tp=tp,
        fp=fp,
        fn=fn,
        strict_tp=strict_tp,
        strict_fp=strict_fp,
        strict_fn=strict_fn,
        group_count=len(groups),
        supported_group_count=supported_groups,
        gold_links_covered_by_groups=covered_by_groups,
        overmerged_group_count=overmerged_groups,
        gold_components=component_count,
        fragmented_gold_components=fragmented_components,
        group_token_tp=group_token_tp,
        group_token_in_groups=group_token_in_groups,
        group_token_total=len(gold_aligned_source) + len(gold_aligned_target),
        unaligned_source_tp=source_unaligned[0],
        unaligned_source_fp=source_unaligned[1],
        unaligned_source_fn=source_unaligned[2],
        unaligned_target_tp=target_unaligned[0],
        unaligned_target_fp=target_unaligned[1],
        unaligned_target_fn=target_unaligned[2],
        confidences=tuple(confidences),
        confidence_correct=tuple(confidence_correct),
        similarities=tuple(similarities),
        origins=tuple(origins),
        group_types=tuple(str(group["type"]) for group in groups),
    )


def _request_payload(
    examples: Sequence[Example],
    *,
    method: str,
    config: str,
    target_language: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_language": "en",
        "target_language": target_language,
        "method": method,
        "sentence_pairs": [
            {
                "id": f"{example.pair}-{example.index}",
                "source": example.source_text,
                "target": example.target_text,
            }
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
    method: str,
    config: str,
    target_language: str,
) -> dict[str, dict[str, Any]]:
    payload = _request_payload(examples, method=method, config=config, target_language=target_language)
    response = client.post(api_url, json=payload)
    response.raise_for_status()
    body = response.json()
    alignments = body.get("sentence_alignments", [])
    if len(alignments) != len(examples):
        raise RuntimeError(f"Expected {len(examples)} alignments, received {len(alignments)}")
    return {str(item["id"]): item for item in alignments}


def _fetch_alignments(
    client: httpx.Client,
    api_url: str,
    examples: Sequence[Example],
    *,
    method: str,
    config: str,
    target_language: str,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    for offset in range(0, len(examples), batch_size):
        batch = list(examples[offset : offset + batch_size])
        try:
            results.update(
                _align_batch(
                    client,
                    api_url,
                    batch,
                    method=method,
                    config=config,
                    target_language=target_language,
                )
            )
        except Exception:
            for example in batch:
                key = f"{example.pair}-{example.index}"
                try:
                    single = _align_batch(
                        client,
                        api_url,
                        [example],
                        method=method,
                        config=config,
                        target_language=target_language,
                    )
                    results.update(single)
                except Exception as exc:  # noqa: BLE001 - batch fallback should not abort the run
                    failures[key] = str(exc)
    return results, failures


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _aggregate(scores: Sequence[SentenceScore]) -> dict[str, Any]:
    tp = sum(score.tp for score in scores)
    fp = sum(score.fp for score in scores)
    fn = sum(score.fn for score in scores)
    strict_tp = sum(score.strict_tp for score in scores)
    strict_fp = sum(score.strict_fp for score in scores)
    strict_fn = sum(score.strict_fn for score in scores)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    strict_precision = _ratio(strict_tp, strict_tp + strict_fp)
    strict_recall = _ratio(strict_tp, strict_tp + strict_fn)
    gold_count = sum(score.gold_count for score in scores)
    group_count = sum(score.group_count for score in scores)
    supported_group_count = sum(score.supported_group_count for score in scores)
    overmerged_group_count = sum(score.overmerged_group_count for score in scores)
    gold_covered = sum(score.gold_links_covered_by_groups for score in scores)
    gold_components = sum(score.gold_components for score in scores)
    fragmented_components = sum(score.fragmented_gold_components for score in scores)
    group_token_tp = sum(score.group_token_tp for score in scores)
    group_token_in_groups = sum(score.group_token_in_groups for score in scores)
    group_token_total = sum(score.group_token_total for score in scores)
    unaligned_source_tp = sum(score.unaligned_source_tp for score in scores)
    unaligned_source_fp = sum(score.unaligned_source_fp for score in scores)
    unaligned_source_fn = sum(score.unaligned_source_fn for score in scores)
    unaligned_target_tp = sum(score.unaligned_target_tp for score in scores)
    unaligned_target_fp = sum(score.unaligned_target_fp for score in scores)
    unaligned_target_fn = sum(score.unaligned_target_fn for score in scores)

    confidences: list[float] = []
    labels: list[bool] = []
    similarities: list[float] = []
    origins: list[str] = []
    for score in scores:
        confidences.extend(score.confidences)
        labels.extend(score.confidence_correct)
        similarities.extend(score.similarities)
        origins.extend(score.origins)

    macro_f1_values = []
    strict_macro_f1_values = []
    for score in scores:
        precision_i = _ratio(score.tp, score.tp + score.fp)
        recall_i = _ratio(score.tp, score.tp + score.fn)
        f1_i = _f1(precision_i, recall_i)
        if f1_i is not None:
            macro_f1_values.append(f1_i)
        strict_precision_i = _ratio(score.strict_tp, score.strict_tp + score.strict_fp)
        strict_recall_i = _ratio(score.strict_tp, score.strict_tp + score.strict_fn)
        strict_f1_i = _f1(strict_precision_i, strict_recall_i)
        if strict_f1_i is not None:
            strict_macro_f1_values.append(strict_f1_i)

    return {
        "sentences": len(scores),
        "gold_links": gold_count,
        "predicted_links": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else None,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": _f1(strict_precision, strict_recall),
        "strict_macro_f1": sum(strict_macro_f1_values) / len(strict_macro_f1_values)
        if strict_macro_f1_values
        else None,
        "group_count": group_count,
        "group_support_precision": _ratio(supported_group_count, group_count),
        "group_gold_coverage_recall": _ratio(gold_covered, gold_count),
        "group_overmerge_rate": _ratio(overmerged_group_count, group_count),
        "group_token_precision": _ratio(group_token_tp, group_token_in_groups),
        "group_token_recall": _ratio(group_token_tp, group_token_total),
        "group_token_f1": _f1(
            _ratio(group_token_tp, group_token_in_groups),
            _ratio(group_token_tp, group_token_total),
        ),
        "gold_components": gold_components,
        "gold_component_fragmentation_rate": _ratio(fragmented_components, gold_components),
        "unaligned_source_f1": _f1(
            _ratio(unaligned_source_tp, unaligned_source_tp + unaligned_source_fp),
            _ratio(unaligned_source_tp, unaligned_source_tp + unaligned_source_fn),
        ),
        "unaligned_target_f1": _f1(
            _ratio(unaligned_target_tp, unaligned_target_tp + unaligned_target_fp),
            _ratio(unaligned_target_tp, unaligned_target_tp + unaligned_target_fn),
        ),
        "confidence_count": len(confidences),
        "confidence_auroc": _auroc(confidences, labels),
        "confidence_ece": _ece(confidences, labels),
        "confidence_brier": _brier(confidences, labels),
        "similarity_auroc": _auroc(similarities, labels),
        "origin_counts": dict(sorted(_count(origins).items())),
        "group_type_counts": dict(sorted(_count(group for score in scores for group in score.group_types).items())),
    }


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = average_rank
        position = end
    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _ece(confidences: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float | None:
    if not confidences or len(confidences) != len(labels):
        return None
    bucket_counts = [0] * bins
    bucket_confidences = [0.0] * bins
    bucket_positives = [0] * bins
    for confidence, label in zip(confidences, labels, strict=True):
        bucket = min(bins - 1, max(0, int(confidence * bins)))
        bucket_counts[bucket] += 1
        bucket_confidences[bucket] += confidence
        bucket_positives[bucket] += int(label)
    total = len(confidences)
    error = 0.0
    for count, confidence_sum, positives in zip(
        bucket_counts, bucket_confidences, bucket_positives, strict=True
    ):
        if count == 0:
            continue
        accuracy = positives / count
        average_confidence = confidence_sum / count
        error += (count / total) * abs(accuracy - average_confidence)
    return error


def _brier(confidences: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not confidences or len(confidences) != len(labels):
        return None
    return sum(
        (confidence - float(label)) ** 2
        for confidence, label in zip(confidences, labels, strict=True)
    ) / len(confidences)


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _bootstrap_f1(scores: Sequence[SentenceScore], iterations: int, seed: int) -> tuple[float | None, float | None]:
    if not scores or iterations <= 0:
        return None, None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [scores[rng.randrange(len(scores))] for _ in scores]
        tp = sum(score.tp for score in sample)
        fp = sum(score.fp for score in sample)
        fn = sum(score.fn for score in sample)
        f1 = _f1(_ratio(tp, tp + fp), _ratio(tp, tp + fn))
        if f1 is not None:
            values.append(f1)
    if not values:
        return None, None
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return lower, upper


def _format(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _print_summary(rows: Sequence[dict[str, Any]]) -> None:
    print(
        "pair  config        n     gold   pred   P      R      F1     F1*    gR     gTokF1 merge  frag   ECE    AUROC"
    )
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['pair']:<5} {row['config']:<13} {metrics['sentences']:>3}  "
            f"{metrics['gold_links']:>5}  {metrics['predicted_links']:>5}  "
            f"{_format(metrics['precision']):<6} {_format(metrics['recall']):<6} "
            f"{_format(metrics['f1']):<6} {_format(metrics['strict_f1']):<6} "
            f"{_format(metrics['group_gold_coverage_recall']):<6} "
            f"{_format(metrics['group_token_f1']):<7} "
            f"{_format(metrics['group_overmerge_rate']):<6} "
            f"{_format(metrics['gold_component_fragmentation_rate']):<6} "
            f"{_format(metrics['confidence_ece']):<6} {_format(metrics['confidence_auroc'])}"
        )


def _dataset_commit() -> str | None:
    repo = XLWA_DATA.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the live word-alignment API on XL-WA gold data.")
    parser.add_argument("--api", default="http://localhost:8000/api/v1/align")
    parser.add_argument("--pairs", nargs="+", default=["it", "es", "nl", "ru"])
    parser.add_argument("--split", default="test", choices=("dev", "test", "train"))
    parser.add_argument("--methods", nargs="+", default=["itermax"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_NAMES), choices=CONFIG_NAMES)
    parser.add_argument("--limit", type=int, default=0, help="Maximum sentences per language pair (0 = all).")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "xlwa_summary.json")
    parser.add_argument("--sentences", type=Path, default=RESULTS_DIR / "xlwa_sentences.jsonl")
    args = parser.parse_args(argv)

    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")

    started_at = datetime.now(timezone.utc)
    all_scores: list[SentenceScore] = []
    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}

    with httpx.Client(timeout=args.timeout) as client:
        for pair in args.pairs:
            examples = _read_examples(pair, args.split, args.limit)
            print(f"Loaded {len(examples)} {pair.upper()}-EN examples from {args.split}.", file=sys.stderr)
            for method in args.methods:
                for config in args.configs:
                    run_started = time.perf_counter()
                    alignments, run_failures = _fetch_alignments(
                        client,
                        args.api,
                        examples,
                        method=method,
                        config=config,
                        target_language=pair,
                        batch_size=args.batch_size,
                    )
                    failures.update(run_failures)
                    scores: list[SentenceScore] = []
                    for example in examples:
                        key = f"{example.pair}-{example.index}"
                        response = alignments.get(key)
                        if response is None:
                            continue
                        scores.append(_score_sentence(example, response, config=config, method=method))
                    all_scores.extend(scores)
                    elapsed = time.perf_counter() - run_started
                    metrics = _aggregate(scores)
                    lower, upper = _bootstrap_f1(scores, args.bootstrap, args.seed + len(rows))
                    metrics["f1_ci95"] = [lower, upper]
                    metrics["elapsed_seconds"] = elapsed
                    metrics["failures"] = len(run_failures)
                    rows.append(
                        {
                            "pair": pair,
                            "method": method,
                            "config": config,
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"{pair} {method} {config}: {len(scores)} sentences, "
                        f"F1={_format(metrics['f1'])}, P={_format(metrics['precision'])}, "
                        f"R={_format(metrics['recall'])}, {elapsed:.1f}s",
                        file=sys.stderr,
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.sentences.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": started_at.isoformat(),
        "api": args.api,
        "dataset": "XL-WA",
        "dataset_commit": _dataset_commit(),
        "split": args.split,
        "pairs": args.pairs,
        "methods": args.methods,
        "configs": args.configs,
        "limit": args.limit,
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "results": rows,
        "failures": failures,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.sentences.open("w", encoding="utf-8") as handle:
        for score in all_scores:
            handle.write(json.dumps(score.__dict__, ensure_ascii=False) + "\n")

    _print_summary(rows)
    print(f"\nWrote {args.output} and {args.sentences}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
