#!/usr/bin/env python3
"""Move an already-verified Baidu raw tree into a superseding release root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from .upload_duplexconv_baidu_release import (
        EXPECTED_CLIENT_SHA256,
        load_jsonl,
        parse_remote_size,
        remote_missing,
        run_client,
        sha256_file,
    )
except ImportError:
    from upload_duplexconv_baidu_release import (
        EXPECTED_CLIENT_SHA256,
        load_jsonl,
        parse_remote_size,
        remote_missing,
        run_client,
        sha256_file,
    )


APPROVED_PREFIX = "/soulx-stage3-dataset-CN/datasets/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-release-root", required=True)
    parser.add_argument("--destination-release-root", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def remote_exists(client: Path, path: str) -> bool:
    result = run_client(client, ["meta", path])
    if result.returncode == 0 and not remote_missing(result.stdout):
        return True
    if remote_missing(result.stdout):
        return False
    raise RuntimeError(f"could not determine remote path state: {path}")


def migration_state(source_exists: bool, destination_exists: bool) -> str:
    if source_exists and not destination_exists:
        return "ready_to_move"
    if not source_exists and destination_exists:
        return "already_moved_verify_resume"
    if source_exists and destination_exists:
        raise RuntimeError("source and destination raw trees both exist; refusing ambiguity")
    raise RuntimeError("both source and destination raw trees are missing")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite migration receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_records(
    client: Path,
    records: list[dict[str, Any]],
    *,
    source_root: str,
    destination_root: str,
) -> int:
    verified_bytes = 0
    prefix = source_root.rstrip("/") + "/"
    for position, record in enumerate(records, start=1):
        old_path = str(record["remote_path"])
        if not old_path.startswith(prefix):
            raise ValueError(f"raw manifest path escapes source raw root: {old_path}")
        relative = old_path[len(prefix) :]
        new_path = str(PurePosixPath(destination_root) / relative)
        print(f"[{position}/{len(records)}] verify migrated raw={new_path}", flush=True)
        meta = run_client(client, ["meta", new_path])
        if meta.returncode != 0 or remote_missing(meta.stdout):
            raise RuntimeError(f"migrated raw file is missing: {new_path}")
        actual = parse_remote_size(meta.stdout)
        expected = int(record["bytes"])
        if actual != expected:
            raise RuntimeError(
                f"migrated raw size mismatch: path={new_path} expected={expected} actual={actual}"
            )
        verified_bytes += expected
    return verified_bytes


def main() -> int:
    args = parse_args()
    client = args.client.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    receipt = args.receipt.resolve()
    source_release_root = str(PurePosixPath(args.source_release_root))
    destination_release_root = str(PurePosixPath(args.destination_release_root))
    for value in (source_release_root, destination_release_root):
        if not value.startswith(APPROVED_PREFIX):
            raise ValueError(f"release root escapes approved dataset prefix: {value}")
    if source_release_root == destination_release_root:
        raise ValueError("source and destination release roots must differ")
    if sha256_file(client) != EXPECTED_CLIENT_SHA256:
        raise ValueError("installed BaiduPCS-Go binary SHA-256 mismatch")

    records = [
        item for item in load_jsonl(source_manifest) if item.get("category") == "raw"
    ]
    if not records:
        raise ValueError("source manifest contains no raw records")
    source_raw = str(PurePosixPath(source_release_root) / "raw")
    destination_raw = str(PurePosixPath(destination_release_root) / "raw")
    source_exists = remote_exists(client, source_raw)
    destination_exists = remote_exists(client, destination_raw)
    state = migration_state(source_exists, destination_exists)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run_passed" if args.dry_run else "pending",
        "checked_at_utc": utc_now(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_release_root": source_release_root,
        "destination_release_root": destination_release_root,
        "source_raw": source_raw,
        "destination_raw": destination_raw,
        "raw_record_count": len(records),
        "raw_payload_bytes": sum(int(item["bytes"]) for item in records),
        "initial_state": state,
        "remote_copy_created": False,
        "rollback_command": f"BaiduPCS-Go mv {destination_raw} {source_release_root}",
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    who = run_client(client, ["who"])
    quota = run_client(client, ["quota"])
    if who.returncode != 0 or quota.returncode != 0:
        raise RuntimeError("Baidu login or quota preflight failed; owner login is required")

    if state == "ready_to_move":
        if not remote_exists(client, destination_release_root):
            made = run_client(client, ["mkdir", destination_release_root])
            if made.returncode != 0 and not remote_exists(client, destination_release_root):
                raise RuntimeError("failed to create destination release root")
        moved = run_client(client, ["mv", source_raw, destination_release_root])
        if moved.returncode != 0:
            raise RuntimeError(f"server-side raw move failed: {moved.stdout}")
        result["migration_action"] = "server_side_move"
    else:
        result["migration_action"] = "resume_verify_preexisting_destination"

    if remote_exists(client, source_raw):
        raise RuntimeError("source raw tree still exists after migration")
    if not remote_exists(client, destination_raw):
        raise RuntimeError("destination raw tree is missing after migration")
    verified_bytes = verify_records(
        client,
        records,
        source_root=source_raw,
        destination_root=destination_raw,
    )
    result.update(
        {
            "status": "passed",
            "completed_at_utc": utc_now(),
            "verified_raw_record_count": len(records),
            "verified_raw_payload_bytes": verified_bytes,
            "source_raw_absent": True,
            "destination_raw_present": True,
        }
    )
    write_json_atomic(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
