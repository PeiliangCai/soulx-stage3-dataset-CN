#!/usr/bin/env python3
"""Build a deterministic compatibility bundle for zero-byte Baidu release files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-manifest", type=Path, required=True)
    parser.add_argument("--restore-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--aggregate-directory-name")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compatibility_record(source: Path, remote_parent: str) -> dict[str, Any]:
    resolved = source.resolve(strict=True)
    stat = resolved.stat()
    return {
        "category": "empty_file_compat",
        "source_path": str(resolved),
        "remote_path": (PurePosixPath(remote_parent) / resolved.name).as_posix(),
        "remote_parent": remote_parent,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def infer_aggregate_directory_name(
    zero_records: list[dict[str, Any]], requested: str | None = None
) -> str:
    if not zero_records:
        raise ValueError("at least one zero-byte record is required")
    discovered: set[str] = set()
    for record in zero_records:
        parts = PurePosixPath(str(record["remote_path"])).parts
        positions = [index for index, part in enumerate(parts) if part == "model_ready"]
        if len(positions) != 1 or positions[0] + 1 >= len(parts):
            raise ValueError(
                f"zero-byte remote path is outside a model_ready aggregate: {record['remote_path']}"
            )
        discovered.add(parts[positions[0] + 1])
    if len(discovered) != 1:
        raise ValueError(f"zero-byte records span multiple aggregates: {sorted(discovered)}")
    aggregate_name = next(iter(discovered))
    if requested is not None and requested != aggregate_name:
        raise ValueError(
            f"requested aggregate directory differs from manifest: "
            f"requested={requested} manifest={aggregate_name}"
        )
    if aggregate_name in {"", ".", ".."} or "/" in aggregate_name:
        raise ValueError(f"unsafe aggregate directory name: {aggregate_name}")
    return aggregate_name


def main() -> int:
    args = parse_args()
    core_manifest = args.core_manifest.resolve(strict=True)
    restore_script = args.restore_script.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    archive_path = output_dir / "empty_files.tar"
    manifest_path = output_dir / "EMPTY_FILES.json"
    supplemental_path = output_dir / "empty_file_compat_manifest.jsonl"
    summary_path = output_dir / "empty_file_compat_summary.json"
    outputs = [archive_path, manifest_path, supplemental_path, summary_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite compatibility outputs: {existing}")

    zero_records = [record for record in load_jsonl(core_manifest) if int(record["bytes"]) == 0]
    aggregate_directory_name = infer_aggregate_directory_name(
        zero_records, args.aggregate_directory_name
    )
    remote_marker = f"/model_ready/{aggregate_directory_name}/"
    files: list[dict[str, str]] = []
    for record in zero_records:
        source = Path(record["source_path"]).resolve(strict=True)
        if source.is_symlink() or not source.is_file() or source.stat().st_size != 0:
            raise ValueError(f"zero-byte source identity changed: {source}")
        if record["sha256"] != EMPTY_SHA256 or sha256_file(source) != EMPTY_SHA256:
            raise ValueError(f"zero-byte source SHA-256 changed: {source}")
        remote_path = str(record["remote_path"])
        if remote_marker not in remote_path:
            raise ValueError(f"zero-byte remote path is outside aggregate root: {remote_path}")
        relative_path = remote_path.split(remote_marker, maxsplit=1)[1]
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"unsafe relative path: {relative_path}")
        files.append(
            {
                "relative_path": pure.as_posix(),
                "sha256": EMPTY_SHA256,
                "original_remote_path": remote_path,
            }
        )
    files.sort(key=lambda item: item["relative_path"])

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for item in files:
            info = tarfile.TarInfo(name=item["relative_path"])
            info.size = 0
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(b""))
    atomic_write(archive_path, archive_buffer.getvalue())
    archive_sha256 = sha256_file(archive_path)

    manifest = {
        "schema_version": 1,
        "profile": "baidu-zero-byte-compatibility-v1",
        "reason": "BaiduPCS-Go v4.0.1 refuses zero-byte local files.",
        "aggregate_directory_name": aggregate_directory_name,
        "empty_file_count": len(files),
        "empty_file_sha256": EMPTY_SHA256,
        "archive_name": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "restore_script_name": restore_script.name,
        "restore_command": (
            "python restore_baidu_empty_release_files.py "
            f"--aggregate-dir /path/to/{aggregate_directory_name} "
            "--manifest EMPTY_FILES.json --archive empty_files.tar"
        ),
        "files": files,
    }
    atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    remote_parent = (
        PurePosixPath(args.remote_root) / "receipts" / "empty_files_compat"
    ).as_posix()
    supplemental_records = [
        compatibility_record(archive_path, remote_parent),
        compatibility_record(manifest_path, remote_parent),
        compatibility_record(restore_script, remote_parent),
    ]
    supplemental_records.sort(key=lambda item: item["remote_path"])
    atomic_write(
        supplemental_path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in supplemental_records
        ).encode("utf-8"),
    )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "zero_byte_record_count": len(files),
        "archive_sha256": archive_sha256,
        "supplemental_record_count": len(supplemental_records),
        "supplemental_payload_bytes": sum(int(item["bytes"]) for item in supplemental_records),
        "supplemental_manifest_sha256": sha256_file(supplemental_path),
    }
    atomic_write(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
