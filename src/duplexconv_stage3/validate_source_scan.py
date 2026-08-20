"""Independent contract validation for a formal DuplexConv source scan."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .source_scan import EXPECTED, SOURCE_VIEW_PROFILE, sha256_file


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON value is not an object")
            yield value


def _validate_checksums(scan_dir: Path) -> dict[str, str]:
    checksum_path = scan_dir / "checksums.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        if filename in expected:
            raise ValueError(f"duplicate checksum entry: {filename}")
        expected[filename] = digest
    for filename, digest in expected.items():
        actual = sha256_file(scan_dir / filename)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {filename}: {actual} != {digest}")
    return expected


def validate_source_scan(scan_dir: Path) -> dict[str, Any]:
    scan_dir = scan_dir.resolve(strict=True)
    checksums = _validate_checksums(scan_dir)
    summary = json.loads((scan_dir / "summary.json").read_text(encoding="utf-8"))
    sources = list(_read_jsonl(scan_dir / "source_inventory.jsonl"))
    events = list(_read_jsonl(scan_dir / "events.jsonl"))
    views = list(_read_jsonl(scan_dir / "target_views.jsonl"))
    quarantines = list(_read_jsonl(scan_dir / "source_quarantine.jsonl"))
    anomalies = list(_read_jsonl(scan_dir / "source_anomalies.jsonl"))

    source_ids = [item["source_id"] for item in sources]
    event_ids = [item["event_id"] for item in events]
    view_ids = [item["view_id"] for item in views]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs are not unique")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event IDs are not unique")
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("view IDs are not unique")

    source_by_id = {item["source_id"]: item for item in sources}
    view_targets_by_source: dict[str, list[int]] = defaultdict(list)
    referenced_event_ids: list[str] = []
    view_counts = Counter()
    for view in views:
        source_id = view["source_id"]
        if source_id not in source_by_id:
            raise ValueError(f"view references missing source: {view['view_id']}")
        ntrack = view["source_ntrack"]
        target_channel = view["target_channel"]
        if view["source_view_profile"] != SOURCE_VIEW_PROFILE:
            raise ValueError(f"wrong source profile: {view['view_id']}")
        if target_channel < 0 or target_channel >= ntrack:
            raise ValueError(f"target channel out of range: {view['view_id']}")
        expected_reference = [channel for channel in range(ntrack) if channel != target_channel]
        if view["reference_channels"] != expected_reference:
            raise ValueError(f"wrong reference channels: {view['view_id']}")
        expected_domain = "multi_party_supplemental" if ntrack == 3 else "two_party"
        if view["conversation_domain"] != expected_domain:
            raise ValueError(f"wrong conversation domain: {view['view_id']}")

        chunk_count = view["chunk_count"]
        target = view["target_active_by_chunk"]
        other = view["other_active_by_chunk"]
        other_count = view["other_active_count_by_chunk"]
        overlap = view["overlap_by_chunk"]
        if not all(
            len(values) == chunk_count
            for values in (target, other, other_count, overlap)
        ):
            raise ValueError(f"activity length mismatch: {view['view_id']}")
        for index in range(chunk_count):
            if other[index] != (other_count[index] > 0):
                raise ValueError(f"other union mismatch: {view['view_id']} chunk {index}")
            if not 0 <= other_count[index] <= ntrack - 1:
                raise ValueError(f"other count out of range: {view['view_id']} chunk {index}")
            if overlap[index] != (target[index] and other[index]):
                raise ValueError(f"overlap mismatch: {view['view_id']} chunk {index}")

        view_targets_by_source[source_id].append(target_channel)
        referenced_event_ids.extend(view["target_event_ids"])
        view_counts[str(ntrack)] += 1

    for source_id, source in source_by_id.items():
        if source["structurally_usable"]:
            expected_targets = list(range(source["ntrack"]))
            actual_targets = sorted(view_targets_by_source[source_id])
            if actual_targets != expected_targets:
                raise ValueError(
                    f"target-vs-rest view closure failed for {source_id}: "
                    f"{actual_targets} != {expected_targets}"
                )

    if Counter(referenced_event_ids) != Counter(event_ids):
        raise ValueError("target views do not cover each event exactly once")
    if source_ids != sorted(source_ids) or event_ids != sorted(event_ids) or view_ids != sorted(view_ids):
        raise ValueError("formal records are not in deterministic ID order")

    assertions = {
        "checksums": True,
        "source_count": len(sources) == EXPECTED["source_count"],
        "target_view_count": len(views) == EXPECTED["target_view_count"],
        "event_count": len(events) == EXPECTED["event_count"],
        "no_source_quarantine": len(quarantines) == 0,
        "view_counts_by_ntrack": dict(sorted(view_counts.items())) == {"2": 990, "3": 15},
        "three_channel_target_vs_rest": all(
            sorted(view_targets_by_source[item["source_id"]]) == [0, 1, 2]
            for item in sources
            if item["ntrack"] == 3
        ),
        "event_exactly_one_target_view": True,
        "activity_arrays": True,
        "summary_gate": summary["gate_3_source_contract_passed"] is True,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise ValueError(f"source scan validation failed: {failed}")
    return {
        "schema_version": 1,
        "status": "passed",
        "scan_dir": str(scan_dir),
        "assertions": assertions,
        "counts": {
            "sources": len(sources),
            "events": len(events),
            "target_views": len(views),
            "views_by_ntrack": dict(sorted(view_counts.items())),
            "source_quarantines": len(quarantines),
            "sources_with_tail_repairs": len(anomalies),
            "checksum_entries": len(checksums),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a formal source_scan_v1 output.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_source_scan(args.scan_dir)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite report: {args.report}")
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
