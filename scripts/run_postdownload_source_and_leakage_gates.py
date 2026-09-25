#!/usr/bin/env python3
"""Wait for a verified shard download, then run source and leakage gates only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--source-scan-output", type=Path, required=True)
    parser.add_argument("--source-scan-log", type=Path, required=True)
    parser.add_argument("--benchmark-records", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--leakage-output", type=Path, required=True)
    parser.add_argument("--leakage-log", type=Path, required=True)
    parser.add_argument("--pipeline-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = utc_now()
    state = {
        "schema_version": 1,
        "profile": "duplexconv-postdownload-source-and-leakage-gates-v1",
        "status": "waiting_for_verified_download",
        "started_at_utc": started,
        "updated_at_utc": started,
        "scope": "source contract scan and frozen benchmark Gate A/B/C only; no Qwen, training, or checkpoint evaluation",
        "download_manifest": str(args.download_manifest),
        "contract": str(args.contract),
    }
    atomic_json(args.pipeline_manifest, state)

    while True:
        if args.download_manifest.exists():
            download = load_json(args.download_manifest)
            status = download.get("status")
            if status in {"complete", "already_complete"}:
                break
            if status == "failed":
                state.update(status="blocked_download_failed", updated_at_utc=utc_now())
                atomic_json(args.pipeline_manifest, state)
                return 20
        time.sleep(args.poll_seconds)

    contract = load_json(args.contract)
    archive = Path(download["output"])
    actual_bytes = download.get("output_bytes", download.get("bytes"))
    actual_sha256 = download.get("output_sha256", download.get("sha256"))
    if not (
        archive.is_file()
        and archive.stat().st_size == contract["official_archive_bytes"] == actual_bytes
        and contract["official_archive_sha256"] == actual_sha256
    ):
        state.update(status="blocked_download_identity_mismatch", updated_at_utc=utc_now())
        atomic_json(args.pipeline_manifest, state)
        return 21

    state.update(
        status="running_source_contract_scan",
        updated_at_utc=utc_now(),
        archive=str(archive),
        archive_bytes=actual_bytes,
        archive_sha256=actual_sha256,
    )
    atomic_json(args.pipeline_manifest, state)
    scan_command = [
        sys.executable,
        str(Path(__file__).with_name("scan_duplexconv_source.py")),
        "--audio-archive",
        str(archive),
        "--metadata-archive",
        str(args.metadata_archive),
        "--output-dir",
        str(args.source_scan_output),
        "--contract-json",
        str(args.contract),
    ]
    scan_returncode = run_logged(scan_command, args.source_scan_log)
    summary_path = args.source_scan_output / "summary.json"
    if scan_returncode != 0 or not summary_path.exists():
        state.update(
            status="blocked_source_scan_failed",
            updated_at_utc=utc_now(),
            source_scan_returncode=scan_returncode,
        )
        atomic_json(args.pipeline_manifest, state)
        return 30
    summary = load_json(summary_path)
    if not (
        summary.get("gate_3_source_contract_passed") is True
        and summary.get("sources", {}).get("quarantined") == 0
    ):
        state.update(status="blocked_source_contract_gate_failed", updated_at_utc=utc_now())
        atomic_json(args.pipeline_manifest, state)
        return 31

    state.update(
        status="running_frozen_leakage_gate_a_b_c",
        updated_at_utc=utc_now(),
        source_scan_summary=str(summary_path),
    )
    atomic_json(args.pipeline_manifest, state)
    leakage_command = [
        sys.executable,
        str(Path(__file__).with_name("run_audio_leakage_gate_v2_2.py")),
        "score-tar",
        "--candidate-tar",
        str(archive),
        "--benchmark-records",
        str(args.benchmark_records),
        "--frozen-config",
        str(args.frozen_config),
        "--output-dir",
        str(args.leakage_output),
    ]
    leakage_returncode = run_logged(leakage_command, args.leakage_log)
    leakage_manifest_path = args.leakage_output / "run_manifest.json"
    if not leakage_manifest_path.exists():
        state.update(
            status="blocked_leakage_scorer_failed",
            updated_at_utc=utc_now(),
            leakage_returncode=leakage_returncode,
        )
        atomic_json(args.pipeline_manifest, state)
        return 40
    leakage = load_json(leakage_manifest_path)
    gate_passed = leakage.get("gate_passed") is True
    state.update(
        status=("complete_all_pre_qwen_gates_passed" if gate_passed else "blocked_leakage_gate_failed"),
        updated_at_utc=utc_now(),
        completed_at_utc=utc_now(),
        source_scan_status="passed",
        leakage_returncode=leakage_returncode,
        leakage_gate_passed=gate_passed,
        leakage_quarantined_source_member_count=leakage.get("quarantined_source_member_count"),
        leakage_manifest=str(leakage_manifest_path),
        qwen_calls=0,
        training=False,
        checkpoint_evaluation=False,
    )
    atomic_json(args.pipeline_manifest, state)
    return 0 if gate_passed else 41


if __name__ == "__main__":
    raise SystemExit(main())
