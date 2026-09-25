#!/usr/bin/env python3
"""Build the allowlisted manifest for the final Baidu receipt bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


ALLOWLIST = (
    "final_release_receipt.json",
    "file_manifest.jsonl",
    "SHA256SUMS.local.tsv",
    "manifest_summary.json",
    "empty_file_compat_manifest.jsonl",
    "empty_file_compat_summary.json",
    "upload_progress.jsonl",
    "upload_completion.json",
    "remote_verification.json",
    "manifest_build.log",
    "upload.log",
    "postupload_verify.log",
)
REMOTE_PARENT = (
    "/soulx-stage3-dataset-CN/datasets/"
    "duplexconv_edu0018_0045_stage3_zh_v1/receipts/final"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-parent", default=REMOTE_PARENT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    release_dir = args.release_dir.resolve(strict=True)
    output = args.output.resolve()
    remote_parent = str(PurePosixPath(args.remote_parent))
    if not remote_parent.startswith("/soulx-stage3-dataset-CN/datasets/") or not remote_parent.endswith(
        "/receipts/final"
    ):
        raise ValueError(f"unsafe final receipt parent: {remote_parent}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite receipt manifest: {output}")

    records: list[dict[str, object]] = []
    for name in ALLOWLIST:
        source = release_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"allowlisted receipt is missing, not regular, or a symlink: {source}")
        stat = source.stat()
        remote_path = str(PurePosixPath(remote_parent) / name)
        records.append(
            {
                "schema_version": 1,
                "category": "final_receipt",
                "source_path": str(source),
                "remote_parent": remote_parent,
                "remote_path": remote_path,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(source),
            }
        )

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": "passed",
                "record_count": len(records),
                "payload_bytes": sum(int(record["bytes"]) for record in records),
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
