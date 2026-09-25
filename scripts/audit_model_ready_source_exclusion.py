#!/usr/bin/env python3
"""Independently audit a whole-source sanitized model-ready export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_checksums(root: Path) -> None:
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def parquet_rows(root: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    paths = sorted((root / "data").glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"exactly one Parquet file required: {root}")
    rows = pq.read_table(paths[0], columns=["index", "sequence"]).to_pylist()
    result = {item["index"]: item["sequence"] for item in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate Parquet indexes: {root}")
    return result


def source_id(record: dict[str, Any]) -> str:
    value = record.get("source_id")
    if isinstance(value, str) and value:
        return value
    return str(record["view_id"]).split("/", 1)[0]


def stable_windows(root: Path) -> dict[tuple[str, tuple[int, int]], tuple[dict[str, Any], str]]:
    rows = parquet_rows(root)
    result = {}
    for item in read_jsonl(root / "metadata" / "windows.jsonl"):
        index = item["index"]
        if index not in rows:
            raise ValueError(f"metadata index absent from Parquet: {index}")
        stable = {key: value for key, value in item.items() if key != "index"}
        key = (item["view_id"], tuple(item["chunk_range"]))
        if key in result:
            raise ValueError(f"duplicate stable window key: {key}")
        result[key] = (stable, rows[index])
    if len(result) != len(rows):
        raise ValueError(f"Parquet/metadata row count differs: {root}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sanitized-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config.resolve(strict=True))
    parent = Path(config["parent_model_ready_dir"]).resolve(strict=True)
    sanitized = args.sanitized_dir.resolve(strict=True)
    report = args.report.absolute()
    if report.exists():
        raise FileExistsError(f"refusing to overwrite report: {report}")
    for item in config["frozen_evidence"]:
        path = Path(item["path"]).resolve(strict=True)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"frozen evidence hash changed: {path}")
    verify_checksums(parent)
    verify_checksums(sanitized)

    excluded = set(config["excluded_source_ids"])
    parent_windows = stable_windows(parent)
    sanitized_windows = stable_windows(sanitized)
    expected_windows = {
        key: value
        for key, value in parent_windows.items()
        if source_id(value[0]) not in excluded
    }
    if expected_windows != sanitized_windows:
        missing = sorted(set(expected_windows) - set(sanitized_windows))[:10]
        extra = sorted(set(sanitized_windows) - set(expected_windows))[:10]
        changed = sorted(
            key
            for key in set(expected_windows) & set(sanitized_windows)
            if expected_windows[key] != sanitized_windows[key]
        )[:10]
        raise ValueError(
            f"sanitized windows differ from exact parent-minus-source set: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    def filtered_quarantine(kind: str) -> list[dict[str, Any]]:
        return [
            item
            for item in read_jsonl(parent / "quarantine" / f"{kind}.jsonl")
            if source_id(item) not in excluded
        ]

    for kind in ("chunks", "views"):
        if filtered_quarantine(kind) != read_jsonl(
            sanitized / "quarantine" / f"{kind}.jsonl"
        ):
            raise ValueError(f"sanitized {kind} quarantine differs from parent-minus-source")

    stats = read_json(sanitized / "stats.json")
    contract = read_json(sanitized / "contract.json")
    expected = config["expected_sanitized"]
    actual = {
        "source_conversation_count": len(
            {source_id(value[0]) for value in sanitized_windows.values()}
        ),
        "source_view_count": len(
            {value[0]["view_id"] for value in sanitized_windows.values()}
        ),
        "row_count": len(sanitized_windows),
        "exported_chunk_count": sum(
            value[0]["chunk_count"] for value in sanitized_windows.values()
        ),
        "quarantined_chunk_count": stats["quarantined_chunk_count"],
        "total_effective_chunk_count": stats["total_effective_chunk_count"],
    }
    if actual != expected:
        raise ValueError(f"sanitized expected-count closure failed: {actual} != {expected}")
    if any(source_id(value[0]) in excluded for value in sanitized_windows.values()):
        raise ValueError("excluded source remains in sanitized windows")
    if contract.get("source_exclusion") != config["expected_exclusion_audit"]:
        raise ValueError("contract source-exclusion audit differs from frozen config")
    if stats.get("source_exclusion") != config["expected_exclusion_audit"]:
        raise ValueError("stats source-exclusion audit differs from frozen config")

    result = {
        "schema_version": 1,
        "profile": "whole-source-model-ready-exclusion-independent-audit-v1",
        "status": "passed",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "parent_model_ready_dir": str(parent),
        "sanitized_model_ready_dir": str(sanitized),
        "excluded_source_ids": sorted(excluded),
        "parent_row_count": len(parent_windows),
        **actual,
        "exact_parent_minus_excluded_windows": True,
        "exact_parent_minus_excluded_quarantines": True,
        "all_checksum_entries_passed": True,
        "sanitized_checksums_sha256": sha256(sanitized / "checksums.sha256"),
        "sanitized_contract_sha256": sha256(sanitized / "contract.json"),
        "sanitized_stats_sha256": sha256(sanitized / "stats.json"),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
