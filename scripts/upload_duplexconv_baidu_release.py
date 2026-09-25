#!/usr/bin/env python3
"""Upload an audited release manifest to Baidu without destructive synchronization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


EXPECTED_CLIENT_SHA256 = "56cb54573c59c98c8cf6deb1b154e4ccb9adec78a8ef9cd41f177fb7a541b715"
PROXY_KEYS = {
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
}
CATEGORY_PRIORITY = {
    "empty_file_compat": -1,
    "release_metadata": 0,
    "aggregate_reports": 1,
    "evidence_file": 2,
    "model_ready": 3,
    "raw": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--supplemental-manifest", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--workers-per-file", type=int, default=4)
    parser.add_argument("--retry", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(value)
    return records


def direct_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_KEYS:
        env.pop(key, None)
    return env


def run_client(
    client: Path,
    args: list[str],
    *,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [str(client), *args]
    if not stream:
        return subprocess.run(
            command,
            env=direct_env(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    process = subprocess.Popen(
        command,
        env=direct_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    captured: list[str] = []
    for chunk in iter(lambda: process.stdout.read(4096), ""):
        captured.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    return_code = process.wait()
    return subprocess.CompletedProcess(command, return_code, "".join(captured), None)


def parse_remote_size(output: str) -> int | None:
    patterns = (
        r"文件大小\s*:\s*([0-9]+)\s*(?:B|字节)?",
        r"文件大小\s+([0-9]+)\s*(?:,|B|字节)",
        r"文件大小[^\n\r]*\(([0-9]+)\s*(?:B|字节)\)",
        r"(?:^|[\n\r])\s*大小\s*:\s*([0-9]+)\s*(?:B|字节)?",
        r"size\s*:\s*([0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def remote_missing(output: str) -> bool:
    lowered = output.lower()
    return (
        "31066" in lowered
        or "文件或目录不存在" in output
        or "file or directory does not exist" in lowered
    )


def append_progress(path: Path, record: dict[str, Any]) -> None:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for value in load_jsonl(path):
        remote_path = str(value["remote_path"])
        completed[remote_path] = value
    return completed


def verify_local_identity(records: Iterable[dict[str, Any]]) -> None:
    seen_remote: set[str] = set()
    for record in records:
        source = Path(record["source_path"])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"source is missing, not a regular file, or a symlink: {source}")
        stat = source.stat()
        if stat.st_size != int(record["bytes"]):
            raise ValueError(f"source size changed after manifest creation: {source}")
        if stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ValueError(f"source mtime changed after manifest creation: {source}")
        remote_path = str(record["remote_path"])
        if not remote_path.startswith("/soulx-stage3-dataset-CN/datasets/"):
            raise ValueError(f"remote path escapes approved root: {remote_path}")
        if remote_path in seen_remote:
            raise ValueError(f"duplicate remote path: {remote_path}")
        seen_remote.add(remote_path)


def validate_compatibility_contract(
    core_records: list[dict[str, Any]], supplemental_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    zero_byte_records = [
        record for record in core_records if int(record["bytes"]) == 0
    ]
    if not zero_byte_records:
        raise ValueError("release requires at least one represented zero-byte record")
    if len(supplemental_records) != 3:
        raise ValueError(
            f"expected 3 empty-file compatibility records, got {len(supplemental_records)}"
        )
    if any(
        record.get("category") != "empty_file_compat"
        for record in supplemental_records
    ):
        raise ValueError("unexpected supplemental manifest category")
    compatibility_archives = [
        record
        for record in supplemental_records
        if PurePosixPath(str(record["remote_path"])).name == "empty_files.tar"
    ]
    if len(compatibility_archives) != 1:
        raise ValueError("supplemental manifest must contain exactly one empty_files.tar")
    return zero_byte_records, compatibility_archives[0]


def remote_parents(records: Iterable[dict[str, Any]]) -> list[str]:
    parents: set[str] = set()
    for record in records:
        parent = PurePosixPath(str(record["remote_parent"]))
        while parent.as_posix() not in {"/", "."}:
            parents.add(parent.as_posix())
            parent = parent.parent
    return sorted(parents, key=lambda value: (value.count("/"), value))


def ensure_remote_directories(client: Path, records: list[dict[str, Any]]) -> None:
    for remote_dir in remote_parents(records):
        meta = run_client(client, ["meta", remote_dir])
        if meta.returncode == 0 and not remote_missing(meta.stdout):
            continue
        created = run_client(client, ["mkdir", remote_dir])
        print(created.stdout, end="" if created.stdout.endswith("\n") else "\n")
        if created.returncode != 0:
            second_meta = run_client(client, ["meta", remote_dir])
            if second_meta.returncode != 0 or remote_missing(second_meta.stdout):
                raise RuntimeError(f"failed to create remote directory: {remote_dir}")


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve(strict=True)
    supplemental_manifest = args.supplemental_manifest.resolve(strict=True)
    client = args.client.resolve(strict=True)
    progress = args.progress.resolve()
    completion = args.completion.resolve()
    if args.workers_per_file < 1 or args.workers_per_file > 8:
        raise ValueError("workers-per-file must be in [1, 8]")
    if args.retry < 1 or args.retry > 50:
        raise ValueError("retry must be in [1, 50]")
    if sha256_file(client) != EXPECTED_CLIENT_SHA256:
        raise ValueError("installed BaiduPCS-Go binary SHA-256 mismatch")

    core_records = load_jsonl(manifest)
    supplemental_records = load_jsonl(supplemental_manifest)
    zero_byte_records, compatibility_archive = validate_compatibility_contract(
        core_records, supplemental_records
    )
    records = [*core_records, *supplemental_records]
    verify_local_identity(records)
    records.sort(
        key=lambda item: (
            CATEGORY_PRIORITY.get(str(item["category"]), 100),
            str(item["remote_path"]),
        )
    )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "record_count": len(records),
                "core_manifest_record_count": len(core_records),
                "supplemental_record_count": len(supplemental_records),
                "represented_zero_byte_record_count": len(zero_byte_records),
                "payload_bytes": sum(int(item["bytes"]) for item in records),
                "progress": str(progress),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    who = run_client(client, ["who"])
    quota = run_client(client, ["quota"])
    if who.returncode != 0 or quota.returncode != 0:
        raise RuntimeError("Baidu login or quota preflight failed; owner login is required")
    print("Baidu login and quota preflight passed; credential material is not logged.", flush=True)

    ensure_remote_directories(client, records)
    completed = load_completed(progress)
    uploaded_or_verified_bytes = 0
    for position, record in enumerate(records, start=1):
        remote_path = str(record["remote_path"])
        local_path = Path(record["source_path"])
        expected_bytes = int(record["bytes"])
        previous = completed.get(remote_path)
        if previous is not None:
            if (
                int(previous["bytes"]) != expected_bytes
                or str(previous["sha256"]) != str(record["sha256"])
            ):
                raise ValueError(f"progress identity conflicts with current manifest: {remote_path}")
            uploaded_or_verified_bytes += expected_bytes
            print(f"[{position}/{len(records)}] progress-skip {remote_path}", flush=True)
            continue

        print(
            f"[{position}/{len(records)}] preflight bytes={expected_bytes} remote={remote_path}",
            flush=True,
        )
        if expected_bytes == 0:
            archive_remote_path = str(compatibility_archive["remote_path"])
            archive_progress = completed.get(archive_remote_path)
            if archive_progress is None:
                raise RuntimeError(
                    "zero-byte record reached before its compatibility archive was uploaded"
                )
            if (
                int(archive_progress["bytes"]) != int(compatibility_archive["bytes"])
                or str(archive_progress["sha256"]) != str(compatibility_archive["sha256"])
            ):
                raise ValueError("compatibility archive progress identity mismatch")
            progress_record = {
                "schema_version": 1,
                "completed_at_utc": utc_now(),
                "category": record["category"],
                "source_path": str(local_path),
                "remote_path": remote_path,
                "bytes": 0,
                "sha256": record["sha256"],
                "status": "represented_by_deterministic_empty_files_compatibility_bundle",
                "representation_remote_path": archive_remote_path,
                "representation_sha256": compatibility_archive["sha256"],
            }
            append_progress(progress, progress_record)
            completed[remote_path] = progress_record
            print(
                f"[{position}/{len(records)}] represented-zero-byte {remote_path}",
                flush=True,
            )
            continue
        remote_meta = run_client(client, ["meta", remote_path])
        if remote_meta.returncode == 0 and not remote_missing(remote_meta.stdout):
            remote_bytes = parse_remote_size(remote_meta.stdout)
            if remote_bytes is None:
                raise RuntimeError(f"remote path exists but exact size could not be parsed: {remote_path}")
            if remote_bytes != expected_bytes:
                raise RuntimeError(
                    f"remote path exists with different size; refusing overwrite: "
                    f"remote={remote_path} expected={expected_bytes} actual={remote_bytes}"
                )
            status = "preexisting_same_size_verified"
        else:
            uploaded = run_client(
                client,
                [
                    "upload",
                    str(local_path),
                    str(record["remote_parent"]),
                    "--policy",
                    "skip",
                    "-p",
                    str(args.workers_per_file),
                    "--retry",
                    str(args.retry),
                ],
                stream=True,
            )
            if uploaded.returncode != 0:
                raise RuntimeError(f"upload failed for: {local_path}")
            verified = run_client(client, ["meta", remote_path])
            if verified.returncode != 0 or remote_missing(verified.stdout):
                raise RuntimeError(f"uploaded path is not visible remotely: {remote_path}")
            remote_bytes = parse_remote_size(verified.stdout)
            if remote_bytes != expected_bytes:
                raise RuntimeError(
                    f"remote size mismatch after upload: remote={remote_path} "
                    f"expected={expected_bytes} actual={remote_bytes}"
                )
            status = "uploaded_and_size_verified"

        progress_record = {
            "schema_version": 1,
            "completed_at_utc": utc_now(),
            "category": record["category"],
            "source_path": str(local_path),
            "remote_path": remote_path,
            "bytes": expected_bytes,
            "sha256": record["sha256"],
            "status": status,
        }
        append_progress(progress, progress_record)
        completed[remote_path] = progress_record
        uploaded_or_verified_bytes += expected_bytes

    progress_sha256 = sha256_file(progress)
    summary = {
        "schema_version": 1,
        "status": "all_nonempty_records_uploaded_or_preexisting_same_size_verified_and_zero_byte_records_represented_by_compatibility_bundle",
        "completed_at_utc": utc_now(),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "supplemental_manifest": str(supplemental_manifest),
        "supplemental_manifest_sha256": sha256_file(supplemental_manifest),
        "progress": str(progress),
        "progress_sha256": progress_sha256,
        "record_count": len(records),
        "core_manifest_record_count": len(core_records),
        "supplemental_record_count": len(supplemental_records),
        "direct_remote_file_count": len(records) - len(zero_byte_records),
        "represented_zero_byte_record_count": len(zero_byte_records),
        "zero_byte_representation_remote_path": compatibility_archive["remote_path"],
        "zero_byte_representation_sha256": compatibility_archive["sha256"],
        "payload_bytes": uploaded_or_verified_bytes,
        "remote_verification": "per-file exact byte size via BaiduPCS-Go meta for all non-empty records; deterministic tar plus manifest for 12 zero-byte records",
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
    }
    temporary = completion.with_name(f".{completion.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, completion)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
