#!/usr/bin/env python3
"""Wait for a shard download and run the immutable pilot gates in order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def update_state(
    manifest_path: Path,
    state: dict[str, Any],
    *,
    status: str,
    stage: str,
    **extra: Any,
) -> None:
    state.update(extra)
    state["status"] = status
    state["stage"] = stage
    state["updated_at_utc"] = utc_now()
    atomic_json(manifest_path, state)


def validate_tar_members(
    archive_path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    source_ids: list[str] = []
    non_file_members = 0
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                non_file_members += 1
                continue
            if member.name.lower().endswith(".wav"):
                source_ids.append(PurePosixPath(member.name).stem)
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("candidate tar contains duplicate WAV source IDs")
    source_ids.sort()
    source_ids_sha256 = hashlib.sha256(
        ("\n".join(source_ids) + "\n").encode("utf-8")
    ).hexdigest()
    checks = {
        "source_count": len(source_ids) == expected["source_count"],
        "first_source_id": source_ids[0] == expected["first_source_id"],
        "last_source_id": source_ids[-1] == expected["last_source_id"],
        "source_ids_sha256": source_ids_sha256 == expected["source_ids_sha256"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"tar membership contract failed: {failed}")
    return {
        "wav_member_count": len(source_ids),
        "non_file_member_count": non_file_members,
        "first_source_id": source_ids[0],
        "last_source_id": source_ids[-1],
        "source_ids_sha256": source_ids_sha256,
        "checks": checks,
    }


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC_ROOT)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--source-scan-output", type=Path, required=True)
    parser.add_argument("--benchmark-records", type=Path, required=True)
    parser.add_argument("--frozen-leakage-config", type=Path, required=True)
    parser.add_argument("--leakage-output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--resume-leakage",
        action="store_true",
        help=(
            "Resume only an interrupted leakage stage recorded by the existing "
            "run manifest. Completed download/archive/source gates are reused, "
            "and interrupted partial files are preserved before recomputation."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract_json)
    state: dict[str, Any] = {
        "schema_version": 1,
        "profile": "duplexconv-pilot-gates-v1",
        "started_at_utc": utc_now(),
        "inputs": {
            "download_manifest": str(args.download_manifest.resolve()),
            "audio_archive": str(args.audio_archive.resolve()),
            "metadata_archive": str(args.metadata_archive.resolve()),
            "contract_json": str(args.contract_json.resolve()),
            "benchmark_records": str(args.benchmark_records.resolve()),
            "frozen_leakage_config": str(args.frozen_leakage_config.resolve()),
        },
        "outputs": {
            "source_scan": str(args.source_scan_output.resolve()),
            "leakage_gate": str(args.leakage_output.resolve()),
        },
        "policy": {
            "stop_on_any_failed_gate": True,
            "llm_calls": False,
            "training": False,
            "checkpoint_evaluation": False,
        },
    }
    update_state(
        args.run_manifest,
        state,
        status="running",
        stage="waiting_for_verified_download",
    )

    while True:
        if not args.download_manifest.exists():
            time.sleep(args.poll_seconds)
            continue
        download = load_json(args.download_manifest)
        download_status = download.get("status")
        update_state(
            args.run_manifest,
            state,
            status="running",
            stage="waiting_for_verified_download",
            download_status=download_status,
            download_updated_at_utc=download.get("updated_at_utc"),
            download_bytes=download.get("downloaded_bytes"),
        )
        if download_status == "complete":
            break
        if download_status in {"failed", "blocked"}:
            raise RuntimeError(f"download ended with status={download_status}")
        time.sleep(args.poll_seconds)

    update_state(
        args.run_manifest,
        state,
        status="running",
        stage="verifying_archive",
    )
    archive_path = args.audio_archive.resolve(strict=True)
    expected_bytes = int(contract["official_archive_bytes"])
    expected_sha256 = contract["official_archive_sha256"]
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"archive byte count mismatch: {actual_bytes} != {expected_bytes}"
        )
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("archive SHA-256 does not match the frozen contract")
    member_check = validate_tar_members(archive_path, contract["expected"])
    state["archive_verification"] = {
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "tar_membership": member_check,
        "passed": True,
    }

    if args.source_scan_output.exists():
        raise FileExistsError(
            f"refusing to overwrite source scan output: {args.source_scan_output}"
        )
    update_state(
        args.run_manifest,
        state,
        status="running",
        stage="source_scan",
    )
    source_log = args.run_manifest.parent / "source_scan.log"
    run_logged(
        [
            str(args.python),
            "-m",
            "duplexconv_stage3.source_scan",
            "--audio-archive",
            str(archive_path),
            "--metadata-archive",
            str(args.metadata_archive),
            "--output-dir",
            str(args.source_scan_output),
            "--contract-json",
            str(args.contract_json),
        ],
        source_log,
    )
    source_summary = load_json(args.source_scan_output / "summary.json")
    if not source_summary.get("gate_3_source_contract_passed"):
        raise RuntimeError("source scan contract gate did not pass")
    state["source_scan"] = {
        "summary": str((args.source_scan_output / "summary.json").resolve()),
        "log": str(source_log.resolve()),
        "gate_3_source_contract_passed": True,
        "source_count": source_summary["sources"]["total"],
        "target_view_count": source_summary["views"]["structurally_usable"],
        "event_count": source_summary["events"]["total"],
    }

    if args.leakage_output.exists():
        raise FileExistsError(
            f"refusing to overwrite leakage output: {args.leakage_output}"
        )
    update_state(
        args.run_manifest,
        state,
        status="running",
        stage="frozen_benchmark_audio_leakage_gate",
    )
    leakage_log = args.run_manifest.parent / "leakage_gate.log"
    run_logged(
        [
            str(args.python),
            str(PROJECT_ROOT / "scripts" / "run_audio_leakage_gate.py"),
            "score-tar",
            "--candidate-tar",
            str(archive_path),
            "--benchmark-records",
            str(args.benchmark_records),
            "--frozen-config",
            str(args.frozen_leakage_config),
            "--output-dir",
            str(args.leakage_output),
        ],
        leakage_log,
    )
    leakage_manifest = load_json(args.leakage_output / "run_manifest.json")
    if not leakage_manifest.get("gate_passed"):
        raise RuntimeError("benchmark audio leakage gate did not pass")
    state["leakage_gate"] = {
        "manifest": str((args.leakage_output / "run_manifest.json").resolve()),
        "log": str(leakage_log.resolve()),
        "gate_passed": True,
        "counts": leakage_manifest["counts"],
        "quarantined_source_member_count": leakage_manifest[
            "quarantined_source_member_count"
        ],
    }
    update_state(
        args.run_manifest,
        state,
        status="complete",
        stage="pilot_gates_complete",
        completed_at_utc=utc_now(),
    )
    return state


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def resume_leakage(args: argparse.Namespace) -> dict[str, Any]:
    state = load_json(args.run_manifest)
    if state.get("profile") != "duplexconv-pilot-gates-v1":
        raise ValueError("existing run manifest has the wrong profile")
    if not state.get("archive_verification", {}).get("passed"):
        raise RuntimeError("archive verification was not completed before interruption")
    if not state.get("source_scan", {}).get("gate_3_source_contract_passed"):
        raise RuntimeError("source scan gate was not completed before interruption")
    if state.get("stage") != "frozen_benchmark_audio_leakage_gate":
        raise RuntimeError(
            f"cannot resume leakage from recorded stage={state.get('stage')!r}"
        )

    archive_path = args.audio_archive.resolve(strict=True)
    recorded_archive = state["archive_verification"]
    if archive_path.stat().st_size != recorded_archive["bytes"]:
        raise RuntimeError("audio archive size changed after verified download")
    if (args.source_scan_output / "summary.json").resolve() != Path(
        state["source_scan"]["summary"]
    ).resolve(strict=True):
        raise RuntimeError("source scan output does not match the existing manifest")

    final_leakage_manifest = args.leakage_output / "run_manifest.json"
    if final_leakage_manifest.exists():
        raise FileExistsError(
            "leakage run manifest already exists; refusing an ambiguous resume"
        )

    interrupted_files: list[dict[str, Any]] = []
    if args.leakage_output.exists():
        timestamp = utc_now().replace(":", "").replace("+00:00", "Z")
        interrupted_dir = (
            args.leakage_output / "interrupted_attempts" / timestamp
        )
        candidates = sorted(
            path
            for path in args.leakage_output.iterdir()
            if path.is_file()
            and path.name
            in {
                "candidate_identity_manifest.jsonl.partial",
                "candidate_matches.jsonl.partial",
                "candidate_identity_manifest.jsonl",
                "candidate_matches.jsonl",
            }
        )
        if candidates:
            interrupted_dir.mkdir(parents=True)
        for path in candidates:
            evidence = {
                "original_path": str(path.resolve()),
                "preserved_path": str((interrupted_dir / path.name).resolve()),
                "bytes": path.stat().st_size,
                "lines": line_count(path),
                "sha256": sha256_file(path),
            }
            path.rename(interrupted_dir / path.name)
            interrupted_files.append(evidence)

    resume_history = state.setdefault("resume_history", [])
    resume_record: dict[str, Any] = {
        "resume_number": len(resume_history) + 1,
        "reason": "container_shutdown_during_frozen_benchmark_audio_leakage_gate",
        "started_at_utc": utc_now(),
        "preserved_interrupted_files": interrupted_files,
        "recomputed_from_candidate_view_zero": True,
        "completed_download_reused": True,
        "completed_archive_verification_reused": True,
        "completed_source_scan_reused": True,
        "llm_calls": False,
    }
    resume_history.append(resume_record)
    update_state(
        args.run_manifest,
        state,
        status="running",
        stage="frozen_benchmark_audio_leakage_gate",
        error=None,
    )

    leakage_log = args.run_manifest.parent / (
        f"leakage_gate_resume_{resume_record['resume_number']:03d}.log"
    )
    run_logged(
        [
            str(args.python),
            str(PROJECT_ROOT / "scripts" / "run_audio_leakage_gate.py"),
            "score-tar",
            "--candidate-tar",
            str(archive_path),
            "--benchmark-records",
            str(args.benchmark_records),
            "--frozen-config",
            str(args.frozen_leakage_config),
            "--output-dir",
            str(args.leakage_output),
        ],
        leakage_log,
    )
    leakage_manifest = load_json(final_leakage_manifest)
    if not leakage_manifest.get("gate_passed"):
        raise RuntimeError("benchmark audio leakage gate did not pass")
    resume_record["completed_at_utc"] = utc_now()
    resume_record["status"] = "complete"
    state["leakage_gate"] = {
        "manifest": str(final_leakage_manifest.resolve()),
        "log": str(leakage_log.resolve()),
        "gate_passed": True,
        "counts": leakage_manifest["counts"],
        "quarantined_source_member_count": leakage_manifest[
            "quarantined_source_member_count"
        ],
    }
    update_state(
        args.run_manifest,
        state,
        status="complete",
        stage="pilot_gates_complete",
        completed_at_utc=utc_now(),
    )
    return state


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume_leakage:
        if not args.run_manifest.exists():
            raise FileNotFoundError(
                f"resume requires an existing run manifest: {args.run_manifest}"
            )
    elif args.run_manifest.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {args.run_manifest}")
    try:
        result = resume_leakage(args) if args.resume_leakage else run(args)
    except Exception as exc:
        if args.run_manifest.exists():
            state = load_json(args.run_manifest)
            update_state(
                args.run_manifest,
                state,
                status="failed",
                stage="stopped_on_failed_gate",
                failed_at_utc=utc_now(),
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
