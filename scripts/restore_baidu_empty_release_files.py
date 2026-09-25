#!/usr/bin/env python3
"""Restore zero-byte files that BaiduPCS-Go cannot represent directly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("compatibility manifest must be a JSON object")
    return value


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*pure.parts)


def main() -> int:
    args = parse_args()
    aggregate_dir = args.aggregate_dir.resolve()
    manifest_path = args.manifest.resolve(strict=True)
    archive_path = args.archive.resolve(strict=True)
    manifest = load_json(manifest_path)
    if manifest.get("profile") != "baidu-zero-byte-compatibility-v1":
        raise ValueError("unexpected compatibility profile")
    if manifest.get("empty_file_sha256") != EMPTY_SHA256:
        raise ValueError("unexpected empty-file SHA-256")
    if sha256_file(archive_path) != manifest.get("archive_sha256"):
        raise ValueError("compatibility archive SHA-256 mismatch")

    expected_paths = [str(item["relative_path"]) for item in manifest["files"]]
    if len(expected_paths) != int(manifest["empty_file_count"]):
        raise ValueError("compatibility manifest count mismatch")
    if len(set(expected_paths)) != len(expected_paths):
        raise ValueError("duplicate empty-file path")

    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        member_names = [member.name for member in members]
        if member_names != expected_paths:
            raise ValueError("archive member list differs from manifest")
        for member in members:
            if not member.isfile() or member.size != 0 or member.issym() or member.islnk():
                raise ValueError(f"invalid empty-file archive member: {member.name}")

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    already_present: list[str] = []
    for value in expected_paths:
        relative = safe_relative_path(value)
        target = aggregate_dir / relative
        lexical_target = Path(os.path.abspath(target))
        if os.path.commonpath([aggregate_dir, lexical_target]) != str(aggregate_dir):
            raise ValueError(f"target escapes aggregate directory: {value}")
        if target.is_symlink():
            raise ValueError(f"refusing to replace symlink: {target}")
        if target.exists():
            if not target.is_file() or target.stat().st_size != 0:
                raise FileExistsError(f"refusing to overwrite non-empty or non-file target: {target}")
            already_present.append(value)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb"):
                pass
            target.chmod(0o644)
            created.append(value)
        if sha256_file(target) != EMPTY_SHA256:
            raise ValueError(f"restored empty-file identity mismatch: {target}")

    result = {
        "status": "passed",
        "aggregate_dir": str(aggregate_dir),
        "empty_file_count": len(expected_paths),
        "created_count": len(created),
        "already_present_count": len(already_present),
        "created": created,
        "already_present": already_present,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
