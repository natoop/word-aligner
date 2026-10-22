from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _pair(identifier: str, source: str, target: str | None = None) -> dict[str, Any]:
    return {"id": identifier, "source": source, "target": target if target is not None else source}


def _short_pair(index: int) -> dict[str, Any]:
    return _pair(
        f"short-{index}",
        f"System number {index} updates the record quickly.",
        f"系统编号{index}快速更新记录。",
    )


def _payload(
    pairs: Sequence[dict[str, Any]],
    *,
    source_language: str = "en",
    target_language: str = "zh-Hans",
    method: str = "itermax",
) -> dict[str, Any]:
    return {
        "source_language": source_language,
        "target_language": target_language,
        "method": method,
        "sentence_pairs": list(pairs),
    }


def _post(
    client: httpx.Client, api_url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, Any, float]:
    started = time.perf_counter()
    response = client.post(api_url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body, elapsed


def _is_client_error(status: int, body: Any) -> bool:
    return status == 422 or (isinstance(body, dict) and body.get("error") == "alignment_failed")


def _check(name: str, ok: bool, detail: str) -> Check:
    return Check(name=name, ok=ok, detail=detail)


def _run_validation_checks(client: httpx.Client, api_url: str, timeout: float) -> list[Check]:
    cases: list[tuple[str, dict[str, Any]]] = [
        ("empty source", _payload([{"id": "empty", "source": "", "target": "目标"}])),
        (
            "whitespace source",
            _payload([{"id": "blank", "source": "   ", "target": "目标"}]),
        ),
        (
            "whitespace target",
            _payload([{"id": "blank-target", "source": "Source", "target": " \t "}]),
        ),
        (
            "invalid language code",
            _payload([_pair("language", "Hello", "你好")], source_language="e"),
        ),
        (
            "10,001 character source",
            _payload([_pair("long-char", "a" * 10_001, "目标")]),
        ),
        (
            "empty sentence_pairs",
            _payload([]),
        ),
        (
            "101 sentence pairs",
            _payload([_short_pair(index) for index in range(101)]),
        ),
    ]
    checks: list[Check] = []
    for name, payload in cases:
        status, body, elapsed = _post(client, api_url, payload, timeout)
        checks.append(
            _check(
                name,
                status == 422,
                f"status={status}, elapsed={elapsed:.3f}s, body={json.dumps(body, ensure_ascii=False)[:240]}",
            )
        )
    return checks


def _run_unicode_checks(client: httpx.Client, api_url: str, timeout: float) -> list[Check]:
    pairs = [
        _pair("unicode-1", "Deploy 🚀 the service.", "部署🚀服务。"),
        _pair("unicode-2", "Map 5→3 and 6→4.", "映射5→3和6→4。"),
    ]
    status, body, elapsed = _post(client, api_url, _payload(pairs), timeout)
    checks = [_check("unicode request", status == 200, f"status={status}, elapsed={elapsed:.3f}s")]
    if status != 200:
        return checks

    alignments = {item["id"]: item for item in body["sentence_alignments"]}
    offset_errors: list[str] = []
    for alignment in alignments.values():
        for side, key in (("source", "source"), ("target", "target")):
            text = alignment[key]
            for token in alignment[f"{side}_tokens"]:
                if text[token["start"] : token["end"]] != token["text"]:
                    offset_errors.append(
                        f"{alignment['id']}.{side}[{token['index']}]: "
                        f"{text[token['start']:token['end']]!r} != {token['text']!r}"
                    )
    checks.append(
        _check(
            "character offsets match token text",
            not offset_errors,
            "all tokens match" if not offset_errors else "; ".join(offset_errors[:5]),
        )
    )

    second = alignments["unicode-2"]
    protected = {
        token["text"]
        for token in second["source_tokens"] + second["target_tokens"]
        if token["is_protected"]
    }
    checks.append(
        _check(
            "protected arrow tokens",
            protected == {"5→3", "6→4"},
            f"protected={sorted(protected)}",
        )
    )
    rule_links = [
        (link["source_index"], link["target_index"])
        for link in second["links"]
        if link["origin"] == "rule"
    ]
    checks.append(
        _check(
            "protected arrow rule links",
            len(rule_links) == 2,
            f"rule_links={rule_links}",
        )
    )
    return checks


def _run_length_checks(client: httpx.Client, api_url: str, timeout: float) -> tuple[list[Check], dict[str, Any]]:
    observations: dict[str, Any] = {}
    checks: list[Check] = []
    for word_count in (500, 510, 520, 550, 600):
        text = " ".join(["test"] * word_count)
        status, body, elapsed = _post(
            client,
            api_url,
            _payload([_pair(f"long-{word_count}", text)], target_language="en"),
            timeout,
        )
        observations[str(word_count)] = {
            "status": status,
            "elapsed_seconds": round(elapsed, 4),
            "error": body.get("error") if isinstance(body, dict) else None,
            "message": body.get("message") if isinstance(body, dict) else None,
        }
        checks.append(
            _check(
                f"{word_count} words",
                status in {200, 500},
                f"status={status}, error={observations[str(word_count)]['error']}, "
                f"elapsed={elapsed:.3f}s",
            )
        )

    short = _short_pair(900)
    long_pair = _pair("long-batch-600", " ".join(["test"] * 600))
    status, body, elapsed = _post(
        client,
        api_url,
        _payload([short, long_pair]),
        timeout,
    )
    observations["mixed_batch"] = {
        "status": status,
        "elapsed_seconds": round(elapsed, 4),
        "body": body,
    }
    checks.append(
        _check(
            "mixed batch with oversized pair fails atomically",
            status == 500 and isinstance(body, dict) and body.get("error") == "alignment_failed",
            f"status={status}, body={json.dumps(body, ensure_ascii=False)[:300]}",
        )
    )

    status, _, elapsed = _post(client, api_url, _payload([short]), timeout)
    observations["short_pair_retry"] = {"status": status, "elapsed_seconds": round(elapsed, 4)}
    checks.append(
        _check(
            "short pair succeeds after batch failure",
            status == 200,
            f"status={status}, elapsed={elapsed:.3f}s",
        )
    )
    return checks, observations


def _timed_batch(
    client: httpx.Client, api_url: str, pairs: Sequence[dict[str, Any]], timeout: float
) -> float:
    status, body, elapsed = _post(client, api_url, _payload(pairs), timeout)
    if status != 200:
        raise RuntimeError(f"Performance request failed with status {status}: {body}")
    alignments = body.get("sentence_alignments", [])
    if len(alignments) != len(pairs):
        raise RuntimeError(f"Expected {len(pairs)} alignments, received {len(alignments)}")
    return elapsed


def _run_performance_checks(
    client: httpx.Client, api_url: str, timeout: float
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    performance: dict[str, Any] = {}
    _timed_batch(client, api_url, [_short_pair(index) for index in range(3)], timeout)

    scaling: dict[str, Any] = {}
    for count in (1, 10, 25, 50, 100):
        pairs = [_short_pair(index) for index in range(count)]
        values = [_timed_batch(client, api_url, pairs, timeout) for _ in range(2)]
        scaling[str(count)] = {
            "median_seconds": round(statistics.median(values), 4),
            "seconds_per_pair": round(statistics.median(values) / count, 6),
        }
    performance["short_pair_scaling"] = scaling
    checks.append(
        _check(
            "100 short pairs succeed",
            scaling["100"]["median_seconds"] > 0,
            f"median={scaling['100']['median_seconds']:.3f}s",
        )
    )

    long_text = " ".join(["service"] * 500)
    long_seconds = _timed_batch(
        client,
        api_url,
        [_pair("long-latency", long_text)],
        timeout,
    )
    performance["long_500_words_seconds"] = round(long_seconds, 4)

    chunks = [[_short_pair(round * 1000 + index) for index in range(25)] for round in range(4)]
    for chunk in chunks:
        _timed_batch(client, api_url, chunk, timeout)

    sequential_times: list[float] = []
    concurrent_times: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        for chunk in chunks:
            _timed_batch(client, api_url, chunk, timeout)
        sequential_times.append(time.perf_counter() - started)

        def send(chunk: Sequence[dict[str, Any]]) -> None:
            _timed_batch(client, api_url, chunk, timeout)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(send, chunk) for chunk in chunks]
            for future in futures:
                future.result()
        concurrent_times.append(time.perf_counter() - started)

    sequential_median = statistics.median(sequential_times)
    concurrent_median = statistics.median(concurrent_times)
    performance["concurrency"] = {
        "requests": len(chunks),
        "pairs_per_request": 25,
        "sequential_seconds": [round(value, 4) for value in sequential_times],
        "concurrent_seconds": [round(value, 4) for value in concurrent_times],
        "sequential_median_seconds": round(sequential_median, 4),
        "concurrent_median_seconds": round(concurrent_median, 4),
        "speedup": round(sequential_median / concurrent_median, 4),
    }
    checks.append(
        _check(
            "four concurrent requests succeed",
            concurrent_median > 0,
            f"sequential={sequential_median:.3f}s, concurrent={concurrent_median:.3f}s, "
            f"speedup={sequential_median / concurrent_median:.2f}x",
        )
    )
    return checks, performance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe robustness and performance of the live alignment API.")
    parser.add_argument("--api", default="http://localhost:8000/api/v1/align")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "robustness.json")
    args = parser.parse_args(argv)

    checks: list[Check] = []
    observations: dict[str, Any] = {}
    started_at = datetime.now(timezone.utc)
    with httpx.Client(
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
    ) as client:
        checks.extend(_run_validation_checks(client, args.api, args.timeout))
        checks.extend(_run_unicode_checks(client, args.api, args.timeout))
        length_checks, length_observations = _run_length_checks(client, args.api, args.timeout)
        checks.extend(length_checks)
        observations["length"] = length_observations
        performance_checks, performance = _run_performance_checks(client, args.api, args.timeout)
        checks.extend(performance_checks)

    summary = {
        "api": args.api,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "all_checks_passed": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
        "observations": observations,
        "performance": performance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for check in checks:
        print(f"[{'PASS' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    print(f"Wrote {args.output}")
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
