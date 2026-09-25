#!/usr/bin/env python3
"""Aggregate frozen per-shard state-label provenance for a contiguous Edu range."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SHARD_PATTERN = re.compile(r"state_labels_edu(\d{4})")
EXPECTED_QWEN_SLUG = "qwen/qwen3-235b-a22b-2507"
EXPECTED_QWEN_NAME = "Qwen3-235B-A22B-Instruct-2507"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite state summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def discover_summaries(root: Path) -> dict[int, Path]:
    candidates = list((root / "processed").glob("state_labels_*/summary.json"))
    candidates.extend((root / "work").glob("final_state_labels_*/summary.json"))
    # Edu_0018 predates parameterized directory names and is intentionally
    # bound to this one frozen path.
    legacy_0018 = root / "work/state_labels_v1/summary.json"
    if legacy_0018.is_file():
        candidates.append(legacy_0018)
    result: dict[int, Path] = {}
    for path in candidates:
        if path == legacy_0018:
            shard = 18
        else:
            match = SHARD_PATTERN.search(str(path))
            if match is None:
                continue
            shard = int(match.group(1))
        if shard in result:
            raise RuntimeError(
                f"multiple state summaries discovered for Edu_{shard:04d}: "
                f"{result[shard]} and {path}"
            )
        result[shard] = path.resolve(strict=True)
    return result


def add_nested_counter(target: Counter[str], value: dict[str, Any]) -> None:
    for key, count in value.items():
        target[str(key)] += int(count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duplexconv-root", type=Path, required=True)
    parser.add_argument("--first-shard", type=int, required=True)
    parser.add_argument("--last-shard", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.duplexconv_root.resolve(strict=True)
    if args.first_shard <= 0 or args.last_shard < args.first_shard:
        raise ValueError("invalid contiguous shard range")
    expected = list(range(args.first_shard, args.last_shard + 1))
    discovered = discover_summaries(root)
    missing = [shard for shard in expected if shard not in discovered]
    if missing:
        raise RuntimeError(f"missing frozen state summaries: {missing}")

    state_sources: Counter[str] = Counter()
    final_states: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    schema_attempts: Counter[str] = Counter()
    qwen_usage: Counter[str] = Counter()
    qwen_event_count = 0
    qwen_request_count = 0
    qwen_accepted_id_count = 0
    qwen_cost = 0.0
    total_events = 0
    inputs = []

    for shard in expected:
        path = discovered[shard]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "passed":
            raise RuntimeError(f"state summary did not pass: {path}")
        event_count = int(payload["event_count"])
        if sum(int(v) for v in payload["state_source_counts"].values()) != event_count:
            raise RuntimeError(f"state-source count mismatch: {path}")
        if sum(int(v) for v in payload["final_state_distribution"].values()) != event_count:
            raise RuntimeError(f"final-state count mismatch: {path}")
        qwen = payload["qwen"]
        if qwen.get("model_api_slug") != EXPECTED_QWEN_SLUG:
            raise RuntimeError(f"Qwen API model identity drift: {path}")
        if qwen.get("model_standard_name") != EXPECTED_QWEN_NAME:
            raise RuntimeError(f"Qwen standard model identity drift: {path}")

        total_events += event_count
        add_nested_counter(state_sources, payload["state_source_counts"])
        add_nested_counter(final_states, payload["final_state_distribution"])
        add_nested_counter(providers, qwen.get("providers", {}))
        add_nested_counter(schema_attempts, qwen.get("schema_attempts", {}))
        route_counts = qwen.get("network_route_policy_counts")
        if route_counts is None:
            route_counts = {qwen.get("network_route_policy", "unknown"): qwen["request_count"]}
        add_nested_counter(routes, route_counts)
        usage = qwen["accepted_response_usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            qwen_usage[key] += int(usage[key])
        qwen_cost += float(usage["cost"])
        qwen_event_count += int(qwen["event_count"])
        qwen_request_count += int(qwen["request_count"])
        qwen_accepted_id_count += int(qwen["accepted_openrouter_id_count"])
        inputs.append(
            {
                "shard_id": f"Edu_{shard:04d}",
                "path": str(path),
                "sha256": sha256_file(path),
                "event_count": event_count,
            }
        )

    if sum(state_sources.values()) != total_events or sum(final_states.values()) != total_events:
        raise RuntimeError("aggregate state counts do not close")
    if state_sources["openrouter_qwen3_235b_a22b_instruct_2507"] != qwen_event_count:
        raise RuntimeError("aggregate Qwen event provenance mismatch")

    payload = {
        "schema_version": 1,
        "status": "passed",
        "profile": "duplexconv-contiguous-state-provenance-aggregate-v1",
        "generated_at_utc": utc_now(),
        "duplexconv_root": str(root),
        "shard_range": [f"Edu_{args.first_shard:04d}", f"Edu_{args.last_shard:04d}"],
        "shard_count": len(expected),
        "event_count": total_events,
        "state_source_counts": dict(sorted(state_sources.items())),
        "final_state_distribution": dict(sorted(final_states.items())),
        "qwen": {
            "model_api_slug": EXPECTED_QWEN_SLUG,
            "model_standard_name": EXPECTED_QWEN_NAME,
            "event_count": qwen_event_count,
            "request_count": qwen_request_count,
            "accepted_openrouter_id_count": qwen_accepted_id_count,
            "accepted_response_usage": {
                **dict(sorted(qwen_usage.items())),
                "cost": qwen_cost,
            },
            "providers": dict(sorted(providers.items())),
            "network_route_policy_counts": dict(sorted(routes.items())),
            "schema_attempts": dict(sorted(schema_attempts.items())),
            "daily_budget_usd_per_execution_day": 10.0,
        },
        "inputs": inputs,
    }
    atomic_json_write(args.output.absolute(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
