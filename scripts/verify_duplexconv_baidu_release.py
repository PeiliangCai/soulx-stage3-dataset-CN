#!/usr/bin/env python3
"""Independently re-query every direct Baidu release file and write an audit report."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from upload_duplexconv_baidu_release import (
    EXPECTED_CLIENT_SHA256,
    load_jsonl,
    parse_remote_size,
    remote_missing,
    run_client,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--supplemental-manifest", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def unique_by_remote_path(
    records: list[dict[str, Any]], *, source_name: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        remote_path = str(record["remote_path"])
        if remote_path in indexed:
            raise ValueError(f"duplicate remote path in {source_name}: {remote_path}")
        indexed[remote_path] = record
    return indexed


def validate_local_identity(records: list[dict[str, Any]]) -> None:
    for record in records:
        source = Path(record["source_path"])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"local source is missing, not regular, or a symlink: {source}")
        stat = source.stat()
        if stat.st_size != int(record["bytes"]):
            raise ValueError(f"local source size changed: {source}")
        if stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ValueError(f"local source mtime changed: {source}")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve(strict=True)
    supplemental_manifest = args.supplemental_manifest.resolve(strict=True)
    client = args.client.resolve(strict=True)
    progress = args.progress.resolve(strict=True)
    completion_path = args.completion.resolve(strict=True)
    output = args.output.resolve()

    if sha256_file(client) != EXPECTED_CLIENT_SHA256:
        raise ValueError("installed BaiduPCS-Go binary SHA-256 mismatch")

    core_records = load_jsonl(manifest)
    supplemental_records = load_jsonl(supplemental_manifest)
    combined_records = [*core_records, *supplemental_records]
    manifest_by_remote = unique_by_remote_path(combined_records, source_name="manifests")
    progress_records = load_jsonl(progress)
    progress_by_remote = unique_by_remote_path(progress_records, source_name="progress")
    completion = load_json_object(completion_path)

    if set(progress_by_remote) != set(manifest_by_remote):
        missing = sorted(set(manifest_by_remote) - set(progress_by_remote))
        extra = sorted(set(progress_by_remote) - set(manifest_by_remote))
        raise ValueError(f"progress path set mismatch: missing={missing} extra={extra}")
    if not core_records or len(supplemental_records) != 3:
        raise ValueError(
            f"unexpected manifest counts: core={len(core_records)} supplemental={len(supplemental_records)}"
        )

    zero_records = [record for record in core_records if int(record["bytes"]) == 0]
    if not zero_records:
        raise ValueError("release requires at least one represented zero-byte record")
    direct_records = [record for record in combined_records if int(record["bytes"]) > 0]
    if not direct_records:
        raise ValueError("release has no directly stored remote files")

    expected_archive_paths = [
        str(record["remote_path"])
        for record in supplemental_records
        if PurePosixPath(str(record["remote_path"])).name == "empty_files.tar"
    ]
    if len(expected_archive_paths) != 1:
        raise ValueError("supplemental manifest does not contain exactly one empty_files.tar")
    archive_remote_path = expected_archive_paths[0]
    archive_sha256 = str(manifest_by_remote[archive_remote_path]["sha256"])

    validate_local_identity(combined_records)
    for remote_path, manifest_record in manifest_by_remote.items():
        progress_record = progress_by_remote[remote_path]
        if int(progress_record["bytes"]) != int(manifest_record["bytes"]):
            raise ValueError(f"progress byte mismatch: {remote_path}")
        if str(progress_record["sha256"]) != str(manifest_record["sha256"]):
            raise ValueError(f"progress SHA-256 mismatch: {remote_path}")
        if int(manifest_record["bytes"]) == 0:
            if progress_record.get("status") != "represented_by_deterministic_empty_files_compatibility_bundle":
                raise ValueError(f"unexpected zero-byte progress status: {remote_path}")
            if progress_record.get("representation_remote_path") != archive_remote_path:
                raise ValueError(f"zero-byte representation path mismatch: {remote_path}")
            if progress_record.get("representation_sha256") != archive_sha256:
                raise ValueError(f"zero-byte representation SHA-256 mismatch: {remote_path}")
        elif progress_record.get("status") not in {
            "uploaded_and_size_verified",
            "preexisting_same_size_verified",
        }:
            raise ValueError(f"unexpected direct-file progress status: {remote_path}")

    expected_payload_bytes = sum(int(record["bytes"]) for record in combined_records)
    expected_progress_sha256 = sha256_file(progress)
    completion_expectations = {
        "manifest_sha256": sha256_file(manifest),
        "supplemental_manifest_sha256": sha256_file(supplemental_manifest),
        "progress_sha256": expected_progress_sha256,
        "record_count": len(combined_records),
        "core_manifest_record_count": len(core_records),
        "supplemental_record_count": len(supplemental_records),
        "direct_remote_file_count": len(direct_records),
        "represented_zero_byte_record_count": len(zero_records),
        "payload_bytes": expected_payload_bytes,
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
    }
    for key, expected in completion_expectations.items():
        if completion.get(key) != expected:
            raise ValueError(
                f"completion field mismatch: {key} expected={expected!r} actual={completion.get(key)!r}"
            )

    who = run_client(client, ["who"])
    quota = run_client(client, ["quota"])
    if who.returncode != 0 or quota.returncode != 0:
        raise RuntimeError("Baidu login or quota preflight failed; owner login is required")

    remote_results: list[dict[str, Any]] = []
    verified_bytes = 0
    for position, record in enumerate(sorted(direct_records, key=lambda item: str(item["remote_path"])), start=1):
        remote_path = str(record["remote_path"])
        expected_bytes = int(record["bytes"])
        print(f"[{position}/{len(direct_records)}] verify remote={remote_path}", flush=True)
        meta = run_client(client, ["meta", remote_path])
        if meta.returncode != 0 or remote_missing(meta.stdout):
            raise RuntimeError(f"direct remote file missing: {remote_path}")
        actual_bytes = parse_remote_size(meta.stdout)
        if actual_bytes is None:
            raise RuntimeError(f"could not parse exact remote size: {remote_path}")
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"remote size mismatch: remote={remote_path} expected={expected_bytes} actual={actual_bytes}"
            )
        verified_bytes += actual_bytes
        remote_results.append(
            {
                "remote_path": remote_path,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "status": "exact_byte_size_verified",
            }
        )

    report = {
        "schema_version": 1,
        "status": "passed",
        "verified_at_utc": utc_now(),
        "verification_scope": "independent_post_upload_BaiduPCS-Go_meta_exact_byte_size_for_every_direct_remote_file",
        "manifest": str(manifest),
        "manifest_sha256": completion_expectations["manifest_sha256"],
        "supplemental_manifest": str(supplemental_manifest),
        "supplemental_manifest_sha256": completion_expectations["supplemental_manifest_sha256"],
        "progress": str(progress),
        "progress_sha256": expected_progress_sha256,
        "completion": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "core_manifest_record_count": len(core_records),
        "supplemental_record_count": len(supplemental_records),
        "direct_remote_file_count": len(direct_records),
        "direct_remote_verified_file_count": len(remote_results),
        "direct_remote_verified_bytes": verified_bytes,
        "represented_zero_byte_record_count": len(zero_records),
        "represented_zero_byte_paths": sorted(str(record["remote_path"]) for record in zero_records),
        "zero_byte_representation_remote_path": archive_remote_path,
        "zero_byte_representation_sha256": archive_sha256,
        "payload_bytes": expected_payload_bytes,
        "category_record_counts": dict(sorted(Counter(str(record["category"]) for record in combined_records).items())),
        "local_manifest_identity_recheck": (
            f"passed_size_and_mtime_for_all_{len(combined_records)}_records"
        ),
        "login_preflight": "passed_existing_protected_config_no_credentials_logged",
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
        "remote_file_results": remote_results,
    }
    write_json_atomic(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "remote_file_results"}, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
