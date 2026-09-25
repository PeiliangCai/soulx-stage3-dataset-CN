#!/usr/bin/env python3
"""Retire exact superseded Baidu release trees after a verified replacement exists."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from upload_duplexconv_baidu_release import (
        EXPECTED_CLIENT_SHA256,
        parse_remote_size,
        remote_missing,
        run_client,
        sha256_file,
    )
except ModuleNotFoundError:  # Imported as scripts.retire_baidu_release_after_verification.
    from scripts.upload_duplexconv_baidu_release import (
        EXPECTED_CLIENT_SHA256,
        parse_remote_size,
        remote_missing,
        run_client,
        sha256_file,
    )


OLD_ROOT = "/soulx-stage3-dataset-CN/datasets/duplexconv_edu0018_0045_stage3_zh_v1"
NEW_ROOT = "/soulx-stage3-dataset-CN/datasets/duplexconv_edu0001_0045_stage3_zh_v2"
RETIRE_NAMES = ("model_ready", "evidence", "receipts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--tombstone", type=Path, required=True)
    parser.add_argument("--predelete-receipt", type=Path, required=True)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--old-root", default=OLD_ROOT)
    parser.add_argument("--new-root", default=NEW_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = load_json_object(path)
        if existing != value:
            raise FileExistsError(f"refusing to overwrite different receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_roots(old_root: str, new_root: str) -> None:
    if old_root != OLD_ROOT or new_root != NEW_ROOT:
        raise ValueError(
            f"retirement roots differ from the owner-approved exact roots: "
            f"old={old_root!r} new={new_root!r}"
        )


def validate_verification_contract(
    verification: dict[str, Any], migration: dict[str, Any], *, new_root: str
) -> None:
    expected_verification = {
        "status": "passed",
        "core_manifest_record_count": 239,
        "supplemental_record_count": 3,
        "direct_remote_file_count": 221,
        "direct_remote_verified_file_count": 221,
        "represented_zero_byte_record_count": 21,
        "payload_bytes": 346099226326,
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
    }
    for key, expected in expected_verification.items():
        if verification.get(key) != expected:
            raise ValueError(
                f"replacement verification contract mismatch: {key} "
                f"expected={expected!r} actual={verification.get(key)!r}"
            )
    remote_results = verification.get("remote_file_results")
    if not isinstance(remote_results, list) or len(remote_results) != 221:
        raise ValueError("replacement verification lacks 221 per-file remote results")
    prefix = new_root.rstrip("/") + "/"
    for result in remote_results:
        remote_path = str(result.get("remote_path", ""))
        if not remote_path.startswith(prefix):
            raise ValueError(f"verified path escapes replacement root: {remote_path}")
        if result.get("status") != "exact_byte_size_verified":
            raise ValueError(f"replacement path was not exact-size verified: {remote_path}")
        if result.get("expected_bytes") != result.get("actual_bytes"):
            raise ValueError(f"replacement path byte mismatch: {remote_path}")

    expected_migration = {
        "status": "passed",
        "source_release_root": OLD_ROOT,
        "destination_release_root": new_root,
        "raw_record_count": 29,
        "verified_raw_record_count": 29,
        "raw_payload_bytes": 211827029597,
        "verified_raw_payload_bytes": 211827029597,
        "remote_copy_created": False,
        "source_raw_absent": True,
        "destination_raw_present": True,
    }
    for key, expected in expected_migration.items():
        if migration.get(key) != expected:
            raise ValueError(
                f"raw migration contract mismatch: {key} "
                f"expected={expected!r} actual={migration.get(key)!r}"
            )


def remote_exists(client: Path, remote_path: str) -> bool:
    result = run_client(client, ["meta", remote_path])
    return result.returncode == 0 and not remote_missing(result.stdout)


def exact_remote_size(client: Path, remote_path: str) -> int:
    result = run_client(client, ["meta", remote_path])
    if result.returncode != 0 or remote_missing(result.stdout):
        raise FileNotFoundError(f"remote path is missing: {remote_path}")
    size = parse_remote_size(result.stdout)
    if size is None:
        raise RuntimeError(f"could not parse exact remote size: {remote_path}")
    return size


def upload_small_file(client: Path, source: Path, remote_parent: str) -> str:
    remote_path = (PurePosixPath(remote_parent) / source.name).as_posix()
    expected_bytes = source.stat().st_size
    if remote_exists(client, remote_path):
        if exact_remote_size(client, remote_path) != expected_bytes:
            raise RuntimeError(f"remote file exists with different size: {remote_path}")
        return remote_path
    made = run_client(client, ["mkdir", remote_parent])
    if made.returncode != 0 and not remote_exists(client, remote_parent):
        raise RuntimeError(f"could not create remote parent: {remote_parent}")
    uploaded = run_client(
        client,
        [
            "upload",
            str(source),
            remote_parent,
            "--policy",
            "skip",
            "-p",
            "1",
            "--retry",
            "10",
        ],
        stream=True,
    )
    if uploaded.returncode != 0:
        raise RuntimeError(f"small-file upload failed: {source}")
    if exact_remote_size(client, remote_path) != expected_bytes:
        raise RuntimeError(f"uploaded small-file size mismatch: {remote_path}")
    return remote_path


def main() -> int:
    args = parse_args()
    validate_roots(args.old_root, args.new_root)
    client = args.client.resolve(strict=True)
    verification_path = args.verification.resolve(strict=True)
    migration_path = args.migration_receipt.resolve(strict=True)
    tombstone = args.tombstone.resolve(strict=True)
    predelete_path = args.predelete_receipt.resolve()
    completion_path = args.completion_receipt.resolve()
    if sha256_file(client) != EXPECTED_CLIENT_SHA256:
        raise ValueError("installed BaiduPCS-Go binary SHA-256 mismatch")
    if tombstone.is_symlink() or not tombstone.is_file() or tombstone.stat().st_size == 0:
        raise ValueError("tombstone must be a non-empty regular file")

    verification = load_json_object(verification_path)
    migration = load_json_object(migration_path)
    validate_verification_contract(verification, migration, new_root=args.new_root)

    who = run_client(client, ["who"])
    quota = run_client(client, ["quota"])
    if who.returncode != 0 or quota.returncode != 0:
        raise RuntimeError("Baidu login or quota preflight failed; owner login is required")

    raw_path = f"{args.old_root}/raw"
    retire_paths = [f"{args.old_root}/{name}" for name in RETIRE_NAMES]
    states = {path: remote_exists(client, path) for path in retire_paths}
    if remote_exists(client, raw_path):
        raise RuntimeError("superseded raw unexpectedly reappeared; refusing retirement")
    if args.dry_run:
        result = {
            "schema_version": 1,
            "status": "dry_run_passed",
            "checked_at_utc": utc_now(),
            "old_root": args.old_root,
            "new_root": args.new_root,
            "raw_absent": True,
            "retire_path_presence": states,
            "verification_sha256": sha256_file(verification_path),
            "migration_receipt_sha256": sha256_file(migration_path),
            "remote_delete_count_if_executed": sum(states.values()),
            "recycle_bin_recoverable": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    tombstone_remote = upload_small_file(client, tombstone, args.old_root)
    fixed_predelete = {
        "status": "authorized_predelete_snapshot",
        "old_root": args.old_root,
        "new_root": args.new_root,
        "verification_sha256": sha256_file(verification_path),
        "migration_receipt_sha256": sha256_file(migration_path),
        "tombstone_remote_path": tombstone_remote,
        "raw_absent": True,
        "retire_paths": retire_paths,
        "recycle_bin_recoverable": True,
    }
    if predelete_path.exists():
        predelete = load_json_object(predelete_path)
        for key, expected in fixed_predelete.items():
            if predelete.get(key) != expected:
                raise ValueError(
                    f"existing predelete receipt mismatch: {key} "
                    f"expected={expected!r} actual={predelete.get(key)!r}"
                )
    else:
        predelete = {
            "schema_version": 1,
            **fixed_predelete,
            "created_at_utc": utc_now(),
            "verification": str(verification_path),
            "migration_receipt": str(migration_path),
            "retire_path_presence": states,
        }
        write_json_atomic(predelete_path, predelete)
    receipt_parent = f"{args.new_root}/receipts/retirement"
    predelete_remote = upload_small_file(client, predelete_path, receipt_parent)

    if completion_path.exists():
        completion = load_json_object(completion_path)
        fixed_completion = {
            "status": "passed",
            "old_root": args.old_root,
            "new_root": args.new_root,
            "raw_absent": True,
            "tombstone_remote_path": tombstone_remote,
            "predelete_receipt_remote_path": predelete_remote,
            "predelete_receipt_sha256": sha256_file(predelete_path),
            "replacement_verification_sha256": sha256_file(verification_path),
            "recycle_bin_recoverable": True,
        }
        for key, expected in fixed_completion.items():
            if completion.get(key) != expected:
                raise ValueError(
                    f"existing completion receipt mismatch: {key} "
                    f"expected={expected!r} actual={completion.get(key)!r}"
                )
        for remote_path in retire_paths:
            if remote_exists(client, remote_path):
                raise RuntimeError(
                    f"completion receipt exists but retired path reappeared: {remote_path}"
                )
        completion_remote = upload_small_file(client, completion_path, receipt_parent)
        if exact_remote_size(client, completion_remote) != completion_path.stat().st_size:
            raise RuntimeError("completion receipt failed resume exact-size verification")
        print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    deleted: list[str] = []
    already_absent: list[str] = []
    for remote_path in retire_paths:
        if not remote_exists(client, remote_path):
            already_absent.append(remote_path)
            continue
        removed = run_client(client, ["rm", remote_path], stream=True)
        if removed.returncode != 0:
            raise RuntimeError(f"remote retirement failed: {remote_path}")
        if remote_exists(client, remote_path):
            raise RuntimeError(f"remote path remains after retirement: {remote_path}")
        deleted.append(remote_path)

    if remote_exists(client, raw_path):
        raise RuntimeError("superseded raw reappeared after retirement")
    if exact_remote_size(client, tombstone_remote) != tombstone.stat().st_size:
        raise RuntimeError("redirect tombstone failed final exact-size verification")
    for remote_path in retire_paths:
        if remote_exists(client, remote_path):
            raise RuntimeError(f"retired path failed final absence check: {remote_path}")

    completion = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "old_root": args.old_root,
        "new_root": args.new_root,
        "deleted_to_recycle_bin": deleted,
        "already_absent_on_resume": already_absent,
        "remote_delete_count": len(deleted),
        "raw_absent": True,
        "tombstone_remote_path": tombstone_remote,
        "tombstone_bytes": tombstone.stat().st_size,
        "predelete_receipt_remote_path": predelete_remote,
        "predelete_receipt_sha256": sha256_file(predelete_path),
        "replacement_verification_sha256": sha256_file(verification_path),
        "recycle_bin_recoverable": True,
    }
    write_json_atomic(completion_path, completion)
    completion_remote = upload_small_file(client, completion_path, receipt_parent)
    if exact_remote_size(client, completion_remote) != completion_path.stat().st_size:
        raise RuntimeError("completion receipt failed final exact-size verification")
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
