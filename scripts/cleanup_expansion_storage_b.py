#!/usr/bin/env python3
"""Audit and execute the owner-approved, exact-target storage cleanup B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXECUTE_TOKEN = "CONFIRM_STORAGE_CLEANUP_B_101G_PLUS_15_TARS"
DEPRECATED_ROOT = Path("/root/autodl-tmp/soulx-duplug-stage3-cn-replacement")
EXPECTED_DEPRECATED_TOP_LEVEL = {"archives", "datasets"}
LARGE_WORK_FILES = {
    "/root/autodl-tmp/dataset/duplexconv/work/structural_sanitize_edu0026_v1/Edu_0026_sanitized_499sources_v1.tar": 8170137600,
    "/root/autodl-tmp/dataset/duplexconv/work/structural_sanitize_edu0039_v1/Edu_0039_sanitized_499sources_v1.tar": 8048936960,
    "/root/autodl-tmp/dataset/duplexconv/work/leakage_sanitize_edu0023_v1/Edu_0023_sanitized_499sources_v1.tar": 7965470720,
    "/root/autodl-tmp/dataset/duplexconv/work/structural_sanitize_edu0030_v1/Edu_0030_sanitized_499sources_v1.tar": 7963473920,
    "/root/autodl-tmp/dataset/duplexconv/work/leakage_sanitize_edu0037_v1/Edu_0037_sanitized_498sources_v1.tar": 7729909760,
    "/root/autodl-tmp/dataset/duplexconv/work/leakage_sanitize_edu0032_v1/Edu_0032_sanitized_499sources_v1.tar": 7690178560,
    "/root/autodl-tmp/dataset/duplexconv/work/leakage_sanitize_edu0035_v1/Edu_0035_sanitized_499sources_v1.tar": 7436410880,
    "/root/autodl-tmp/dataset/duplexconv/work/leakage_sanitize_edu0043_v1/Edu_0043_sanitized_499sources_v1.tar": 7432970240,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0041_v1/candidate_model_ready_views.tar": 2709524480,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0044_v1/candidate_model_ready_views.tar": 2607134720,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0040_v1/candidate_model_ready_views.tar": 2509926400,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0043_sanitized_v1/candidate_model_ready_views.tar": 2480691200,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0042_v1/candidate_model_ready_views.tar": 2466170880,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0018_v1/candidate_model_ready_views.tar": 2442065920,
    "/root/autodl-tmp/dataset/duplexconv/work/gate_d_edu0018_sanitized_v1/candidate_model_ready_views.tar": 2440714240,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-token", default="")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def directory_inventory(root: Path) -> dict[str, Any]:
    file_count = 0
    directory_count = 0
    symlink_count = 0
    apparent_bytes = root.lstat().st_size
    top_level: dict[str, dict[str, int]] = {}
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        top_level[entry.name] = {"file_count": 0, "directory_count": 0, "symlink_count": 0, "apparent_bytes": 0}
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path != root:
            directory_count += 1
            apparent_bytes += current_path.lstat().st_size
        relative = current_path.relative_to(root)
        top_name = relative.parts[0] if relative.parts else None
        for name in list(directories):
            path = current_path / name
            if path.is_symlink():
                symlink_count += 1
                apparent_bytes += path.lstat().st_size
                directories.remove(name)
                if top_name or name in top_level:
                    key = top_name or name
                    top_level[key]["symlink_count"] += 1
                    top_level[key]["apparent_bytes"] += path.lstat().st_size
        for name in files:
            path = current_path / name
            stat = path.lstat()
            if path.is_symlink():
                symlink_count += 1
            else:
                file_count += 1
            apparent_bytes += stat.st_size
            key = top_name or name
            if key in top_level:
                top_level[key]["symlink_count" if path.is_symlink() else "file_count"] += 1
                top_level[key]["apparent_bytes"] += stat.st_size
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "apparent_bytes": apparent_bytes,
        "top_level": top_level,
    }


def validate_targets() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if DEPRECATED_ROOT.is_symlink() or not DEPRECATED_ROOT.is_dir():
        raise ValueError(f"deprecated root is missing, not a directory, or a symlink: {DEPRECATED_ROOT}")
    if DEPRECATED_ROOT.resolve(strict=True) != DEPRECATED_ROOT:
        raise ValueError("deprecated root does not resolve to its exact approved path")
    if os.path.ismount(DEPRECATED_ROOT):
        raise ValueError("deprecated root is a mount point; refusing deletion")
    actual_top = {entry.name for entry in DEPRECATED_ROOT.iterdir()}
    if actual_top != EXPECTED_DEPRECATED_TOP_LEVEL:
        raise ValueError(f"deprecated root top-level entries changed: {sorted(actual_top)}")
    inventory = directory_inventory(DEPRECATED_ROOT)
    if not (95 * 2**30 <= int(inventory["apparent_bytes"]) <= 105 * 2**30):
        raise ValueError(f"deprecated root apparent size is outside approved bounds: {inventory['apparent_bytes']}")

    work_records: list[dict[str, Any]] = []
    for value, expected_bytes in LARGE_WORK_FILES.items():
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"approved work tar is missing, not regular, or a symlink: {path}")
        if path.resolve(strict=True) != path:
            raise ValueError(f"work tar does not resolve to its exact approved path: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"work tar size changed: path={path} expected={expected_bytes} actual={actual_bytes}"
            )
        print(f"hashing {path}", flush=True)
        work_records.append(
            {
                "path": str(path),
                "bytes": actual_bytes,
                "sha256": sha256_file(path),
                "role": "rebuildable_sanitized_source_or_gate_d_candidate_tar_not_used_by_training",
            }
        )
    return inventory, work_records


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    predelete_path = output_dir / "storage_cleanup_b_predelete.json"
    completion_path = output_dir / "storage_cleanup_b_completion.json"
    failure_path = output_dir / "storage_cleanup_b_failure.json"
    for path in (predelete_path, completion_path, failure_path):
        if path.exists():
            raise FileExistsError(f"cleanup audit output already exists: {path}")

    disk_before = shutil.disk_usage("/root/autodl-tmp")
    deprecated_inventory, work_records = validate_targets()
    predelete = {
        "schema_version": 1,
        "status": "predelete_audit_passed",
        "audited_at_utc": utc_now(),
        "owner_approved_plan": "storage_cleanup_B",
        "deprecated_root": str(DEPRECATED_ROOT),
        "deprecated_root_inventory": deprecated_inventory,
        "work_tar_count": len(work_records),
        "work_tar_bytes": sum(int(record["bytes"]) for record in work_records),
        "work_tars": work_records,
        "preserved": [
            "/root/autodl-tmp/dataset/duplexconv/raw",
            "/root/autodl-tmp/dataset/duplexconv/processed",
            "/root/autodl-tmp/dataset/duplexconv/model_ready",
            "/root/autodl-tmp/dataset/duplexconv/aggregates",
            "/root/autodl-tmp/dataset/duplexconv/reports",
            "/root/autodl-tmp/dataset/duplexconv/cache",
            "all non-targeted work manifests, request/results, source scans, selection lists, checksums, and reports",
        ],
        "training_dependency_check": "passed_no_aggregate_or_training_path_references_targeted_tars",
        "disk_before": {
            "total_bytes": disk_before.total,
            "used_bytes": disk_before.used,
            "free_bytes": disk_before.free,
        },
    }
    write_json_once(predelete_path, predelete)
    if args.execute_token != EXECUTE_TOKEN:
        print(json.dumps(predelete, ensure_ascii=False, indent=2, sort_keys=True))
        print("audit-only: execute token not supplied", flush=True)
        return 0

    deleted: list[str] = []
    try:
        for record in work_records:
            path = Path(str(record["path"]))
            path.unlink()
            deleted.append(str(path))
        shutil.rmtree(DEPRECATED_ROOT)
        deleted.append(str(DEPRECATED_ROOT))
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "partial_cleanup_failure",
            "failed_at_utc": utc_now(),
            "error": repr(exc),
            "deleted_before_failure": deleted,
        }
        write_json_once(failure_path, failure)
        raise

    remaining = [record["path"] for record in work_records if Path(str(record["path"])).exists()]
    if DEPRECATED_ROOT.exists() or remaining:
        raise RuntimeError(
            f"post-delete absence check failed: deprecated_root={DEPRECATED_ROOT.exists()} remaining={remaining}"
        )
    disk_after = shutil.disk_usage("/root/autodl-tmp")
    completion = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "deleted_deprecated_root": str(DEPRECATED_ROOT),
        "deleted_work_tar_count": len(work_records),
        "deleted_work_tar_bytes": sum(int(record["bytes"]) for record in work_records),
        "deleted_target_count_including_deprecated_root": len(deleted),
        "disk_after": {
            "total_bytes": disk_after.total,
            "used_bytes": disk_after.used,
            "free_bytes": disk_after.free,
            "free_bytes_increase": disk_after.free - disk_before.free,
        },
        "recovery": "Deprecated datasets require redownload. Work tars are deterministically rebuildable from preserved official raw tars, exclusion configs, model-ready selection metadata, and frozen Gate-D inputs.",
    }
    write_json_once(completion_path, completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
