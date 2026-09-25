#!/usr/bin/env python3
"""Upload a small audited receipt bundle without overwriting remote files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload_duplexconv_baidu_release import (
    EXPECTED_CLIENT_SHA256,
    append_progress,
    ensure_remote_directories,
    load_completed,
    load_jsonl,
    parse_remote_size,
    remote_missing,
    run_client,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completion: {path}")
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
    client = args.client.resolve(strict=True)
    progress = args.progress.resolve()
    completion = args.completion.resolve()
    if sha256_file(client) != EXPECTED_CLIENT_SHA256:
        raise ValueError("installed BaiduPCS-Go binary SHA-256 mismatch")

    records = load_jsonl(manifest)
    if not records:
        raise ValueError("receipt manifest is empty")
    seen: set[str] = set()
    approved_parent: str | None = None
    for record in records:
        source = Path(record["source_path"])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"receipt source is missing, not regular, or a symlink: {source}")
        stat = source.stat()
        if stat.st_size != int(record["bytes"]):
            raise ValueError(f"receipt source size changed: {source}")
        if stat.st_mtime_ns != int(record["mtime_ns"]):
            raise ValueError(f"receipt source mtime changed: {source}")
        if sha256_file(source) != str(record["sha256"]):
            raise ValueError(f"receipt source SHA-256 changed: {source}")
        remote_path = str(record["remote_path"])
        remote_parent = str(record["remote_parent"])
        if approved_parent is None:
            approved_parent = remote_parent
        if remote_parent != approved_parent:
            raise ValueError("receipt manifest spans multiple remote parents")
        expected_prefix = remote_parent.rstrip("/") + "/"
        if (
            not remote_parent.startswith("/soulx-stage3-dataset-CN/datasets/")
            or not remote_parent.endswith("/receipts/final")
            or not remote_path.startswith(expected_prefix)
        ):
            raise ValueError(f"receipt path escapes approved final directory: {remote_path}")
        if remote_path in seen:
            raise ValueError(f"duplicate receipt remote path: {remote_path}")
        seen.add(remote_path)

    who = run_client(client, ["who"])
    quota = run_client(client, ["quota"])
    if who.returncode != 0 or quota.returncode != 0:
        raise RuntimeError("Baidu login or quota preflight failed; owner login is required")
    ensure_remote_directories(client, records)

    completed = load_completed(progress)
    for position, record in enumerate(records, start=1):
        remote_path = str(record["remote_path"])
        expected_bytes = int(record["bytes"])
        previous = completed.get(remote_path)
        if previous is not None:
            if (
                int(previous["bytes"]) != expected_bytes
                or str(previous["sha256"]) != str(record["sha256"])
            ):
                raise ValueError(f"receipt progress identity mismatch: {remote_path}")
            print(f"[{position}/{len(records)}] progress-skip {remote_path}", flush=True)
            continue

        meta = run_client(client, ["meta", remote_path])
        if meta.returncode == 0 and not remote_missing(meta.stdout):
            actual_bytes = parse_remote_size(meta.stdout)
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"remote receipt exists with different size; refusing overwrite: "
                    f"remote={remote_path} expected={expected_bytes} actual={actual_bytes}"
                )
            status = "preexisting_same_size_verified"
        else:
            uploaded = run_client(
                client,
                [
                    "upload",
                    str(record["source_path"]),
                    str(record["remote_parent"]),
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
                raise RuntimeError(f"receipt upload failed: {record['source_path']}")
            verified = run_client(client, ["meta", remote_path])
            if verified.returncode != 0 or remote_missing(verified.stdout):
                raise RuntimeError(f"uploaded receipt is missing remotely: {remote_path}")
            actual_bytes = parse_remote_size(verified.stdout)
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"receipt size mismatch: remote={remote_path} "
                    f"expected={expected_bytes} actual={actual_bytes}"
                )
            status = "uploaded_and_size_verified"

        progress_record = {
            "schema_version": 1,
            "completed_at_utc": utc_now(),
            "remote_path": remote_path,
            "source_path": str(record["source_path"]),
            "bytes": expected_bytes,
            "sha256": record["sha256"],
            "status": status,
        }
        append_progress(progress, progress_record)
        completed[remote_path] = progress_record
        print(f"[{position}/{len(records)}] {status} {remote_path}", flush=True)

    summary = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "progress": str(progress),
        "progress_sha256": sha256_file(progress),
        "record_count": len(records),
        "payload_bytes": sum(int(record["bytes"]) for record in records),
        "remote_exact_byte_verification_count": len(records),
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
    }
    write_json_atomic(completion, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
