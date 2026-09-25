#!/usr/bin/env python3
"""Resume-safe direct shard download with disk and identity gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def partial_progress_bytes(partial: Path, aria2_control: Path) -> int:
    if not partial.exists():
        return 0
    if aria2_control.exists():
        # A segmented aria2 transfer can make a sparse file's logical size look
        # nearly complete while most extents are still holes. Allocated blocks
        # are only an approximate progress value, but they are conservative for
        # disk gating and cannot be mistaken for a complete byte stream.
        return partial.stat().st_blocks * 512
    return partial.stat().st_size


def download(args: argparse.Namespace) -> dict:
    destination = args.output.absolute()
    partial = destination.with_suffix(destination.suffix + ".part")
    aria2_control = Path(f"{partial}.aria2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != args.expected_bytes:
            raise RuntimeError("existing destination has the wrong size")
        if sha256_file(destination) != args.expected_sha256:
            raise RuntimeError("existing destination has the wrong SHA-256")
        result = {
            "schema_version": 1,
            "status": "already_complete",
            "completed_at_utc": utc_now(),
            "url": args.url,
            "output": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": args.expected_sha256,
            "network_route": args.network_route,
        }
        write_manifest(args.manifest, result)
        return result

    route_name = args.network_route.lower()
    direct_route = "direct" in route_name
    use_aria2c = "aria2c" in route_name and shutil.which("aria2c") is not None
    transfer_tool = "aria2c" if use_aria2c else "curl"
    if aria2_control.exists() and not use_aria2c:
        raise RuntimeError("an aria2 control file exists but this route does not use aria2c")
    partial_logical_bytes = partial.stat().st_size if partial.exists() else 0
    partial_allocated_bytes = (
        partial.stat().st_blocks * 512 if partial.exists() else 0
    )
    existing = partial_progress_bytes(partial, aria2_control)
    if partial_logical_bytes > args.expected_bytes:
        raise RuntimeError("partial file is larger than the official object")
    usage = shutil.disk_usage(destination.parent)
    remaining = args.expected_bytes - existing
    if usage.free < remaining + args.minimum_free_bytes:
        raise RuntimeError(
            f"disk gate failed: free={usage.free}, remaining={remaining}, "
            f"minimum_free={args.minimum_free_bytes}"
        )
    started = utc_now()
    process_env = os.environ.copy()
    removed_proxy_env_names: list[str] = []
    if direct_route:
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            if name in process_env:
                removed_proxy_env_names.append(name)
                process_env.pop(name)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started,
        "updated_at_utc": started,
        "url": args.url,
        "output": str(destination),
        "partial": str(partial),
        "expected_bytes": args.expected_bytes,
        "expected_sha256": args.expected_sha256,
        "existing_partial_bytes": existing,
        "existing_partial_logical_bytes": partial_logical_bytes,
        "existing_partial_allocated_bytes": partial_allocated_bytes,
        "aria2_control_present_at_start": aria2_control.exists(),
        "downloaded_bytes": existing,
        "minimum_free_bytes": args.minimum_free_bytes,
        "available_bytes": usage.free,
        "network_route": args.network_route,
        "transfer_tool": transfer_tool,
        "parallel_connections": 16 if use_aria2c else 1,
        "proxy_env_inherited": not direct_route,
        "removed_proxy_env_names": sorted(removed_proxy_env_names),
        "second_full_cache_copy": False,
    }
    write_manifest(args.manifest, manifest)
    try:
        partial_is_complete = (
            partial.exists()
            and partial.stat().st_size == args.expected_bytes
            and not aria2_control.exists()
        )
        if not partial_is_complete:
            if use_aria2c:
                command = [
                    "aria2c",
                    "--continue=true",
                    "--max-connection-per-server=16",
                    "--split=16",
                    "--min-split-size=1M",
                    "--file-allocation=none",
                    "--auto-file-renaming=false",
                    "--allow-overwrite=true",
                    "--connect-timeout=20",
                    "--timeout=30",
                    "--max-tries=8",
                    "--retry-wait=2",
                    "--summary-interval=10",
                    f"--dir={partial.parent}",
                    f"--out={partial.name}",
                    args.url,
                ]
            else:
                command = [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "0",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    args.url,
                ]
            process = subprocess.Popen(command, env=process_env)
            while True:
                try:
                    return_code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    usage = shutil.disk_usage(destination.parent)
                    manifest.update(
                        {
                            "updated_at_utc": utc_now(),
                            "downloaded_bytes": partial_progress_bytes(
                                partial, aria2_control
                            ),
                            "partial_logical_bytes": partial.stat().st_size
                            if partial.exists()
                            else 0,
                            "available_bytes": usage.free,
                        }
                    )
                    write_manifest(args.manifest, manifest)
            if return_code != 0:
                raise RuntimeError(f"curl exited with status {return_code}")
        actual_bytes = partial.stat().st_size
        if actual_bytes != args.expected_bytes:
            raise RuntimeError(
                f"download size mismatch: {actual_bytes} != {args.expected_bytes}"
            )
        manifest.update(
            {
                "status": "verifying_sha256",
                "updated_at_utc": utc_now(),
                "downloaded_bytes": actual_bytes,
            }
        )
        write_manifest(args.manifest, manifest)
        actual_sha256 = sha256_file(partial)
        if actual_sha256 != args.expected_sha256:
            raise RuntimeError(
                f"download SHA-256 mismatch: {actual_sha256} != {args.expected_sha256}"
            )
        partial.replace(destination)
        result = {
            **manifest,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "output_bytes": destination.stat().st_size,
            "output_sha256": actual_sha256,
            "available_bytes_after": shutil.disk_usage(destination.parent).free,
        }
        write_manifest(args.manifest, result)
        return result
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "downloaded_bytes": partial_progress_bytes(partial, aria2_control),
                "partial_logical_bytes": partial.stat().st_size
                if partial.exists()
                else 0,
                "available_bytes": shutil.disk_usage(destination.parent).free,
            }
        )
        write_manifest(args.manifest, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--network-route", required=True)
    parser.add_argument(
        "--minimum-free-bytes", type=int, default=20 * 1024 * 1024 * 1024
    )
    return parser


def main() -> int:
    result = download(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
