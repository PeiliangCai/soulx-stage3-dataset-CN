#!/usr/bin/env python3
"""Build a deterministic, credential-safe manifest for a Baidu dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORBIDDEN_NAME_PARTS = {
    ".env",
    "bduss",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "stoken",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"release payload must not contain symlinks: {path}")
        if path.is_file():
            yield path


def assert_safe_name(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & FORBIDDEN_NAME_PARTS:
        raise ValueError(f"credential-like path is forbidden from release: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def remote_join(*parts: str) -> str:
    normalized = PurePosixPath("/")
    for part in parts:
        normalized /= str(part).lstrip("/")
    return normalized.as_posix()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    spec = load_json(args.spec.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir != args.spec.resolve().parent:
        raise ValueError("output directory must be the directory containing release_spec.json")

    output_paths = {
        "jsonl": output_dir / "file_manifest.jsonl",
        "checksums": output_dir / "SHA256SUMS.local.tsv",
        "summary": output_dir / "manifest_summary.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing manifest outputs: {existing}")

    excluded_roots = [Path(value).resolve() for value in spec["excluded_local_roots"]]
    records: list[dict[str, Any]] = []
    seen_remote_paths: set[str] = set()

    def add_file(category: str, source: Path, remote_path: str) -> None:
        resolved = source.resolve(strict=True)
        assert_safe_name(resolved)
        for excluded in excluded_roots:
            if resolved == excluded or excluded in resolved.parents:
                raise ValueError(f"allowlisted file falls below excluded root: {resolved}")
        if remote_path in seen_remote_paths:
            raise ValueError(f"duplicate remote path: {remote_path}")
        seen_remote_paths.add(remote_path)
        stat = resolved.stat()
        print(
            f"hashing category={category} bytes={stat.st_size} path={resolved}",
            file=sys.stderr,
            flush=True,
        )
        records.append(
            {
                "category": category,
                "source_path": str(resolved),
                "remote_path": remote_path,
                "remote_parent": str(PurePosixPath(remote_path).parent),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(resolved),
            }
        )

    for payload in spec["payloads"]:
        local_root = Path(payload["local_path"]).resolve(strict=True)
        files = list(iter_files(local_root))
        payload_bytes = sum(path.stat().st_size for path in files)
        if len(files) != int(payload["expected_file_count"]):
            raise ValueError(
                f"file count mismatch for {payload['name']}: "
                f"expected={payload['expected_file_count']} actual={len(files)}"
            )
        if payload_bytes != int(payload["expected_payload_bytes"]):
            raise ValueError(
                f"payload byte mismatch for {payload['name']}: "
                f"expected={payload['expected_payload_bytes']} actual={payload_bytes}"
            )
        for source in files:
            relative = source.relative_to(local_root).as_posix()
            remote_path = remote_join(
                payload["remote_parent"], local_root.name, relative
            )
            add_file(str(payload["name"]), source, remote_path)

    remote_root = str(spec["remote_root"])
    for value in spec["evidence_files"]:
        source = Path(value)
        add_file("evidence_file", source, remote_join(remote_root, "evidence", source.name))

    for name in ("README.md", "release_spec.json"):
        source = output_dir / name
        add_file("release_metadata", source, remote_join(remote_root, "receipts", name))

    records.sort(key=lambda item: item["remote_path"])
    jsonl = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    checksums = "".join(
        f"{record['sha256']}\t{record['bytes']}\t{record['source_path']}\t{record['remote_path']}\n"
        for record in records
    )
    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    for record in records:
        category = str(record["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        category_bytes[category] = category_bytes.get(category, 0) + int(record["bytes"])
    summary = {
        "schema_version": 1,
        "release_id": spec["release_id"],
        "remote_root": remote_root,
        "record_count": len(records),
        "payload_bytes": sum(int(record["bytes"]) for record in records),
        "category_counts": category_counts,
        "category_bytes": category_bytes,
        "manifest_profile": "absolute-local-source-to-absolute-baidu-remote-v1",
        "credential_like_path_count": 0,
        "remote_path_collision_count": 0,
    }

    atomic_write_text(output_paths["jsonl"], jsonl)
    atomic_write_text(output_paths["checksums"], checksums)
    atomic_write_text(
        output_paths["summary"],
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
