#!/usr/bin/env python3
"""Build and independently audit a parent-minus-one-source audio tar.

The parent archive is immutable.  The output keeps every remaining tar member
payload and the audited identity fields unchanged, while excluding exactly one
complete source member.  Both archives are reread after construction so the
audit does not rely on the write loop's assumptions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, BinaryIO


PROFILE = "duplexconv-parent-minus-one-full-payload-audit-v1"
IDENTITY_FIELDS = (
    "name",
    "size",
    "mode",
    "uid",
    "gid",
    "mtime",
    "uname",
    "gname",
    "type",
    "payload_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def type_text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return value


def inventory(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                raise RuntimeError(f"non-regular tar member is not supported: {member.name}")
            if not member.name.lower().endswith(".wav"):
                raise RuntimeError(f"non-WAV tar member is not supported: {member.name}")
            if member.name in records:
                raise RuntimeError(f"duplicate tar member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read tar member: {member.name}")
            records[member.name] = {
                "name": member.name,
                "size": int(member.size),
                "mode": int(member.mode),
                "uid": int(member.uid),
                "gid": int(member.gid),
                "mtime": member.mtime,
                "uname": member.uname,
                "gname": member.gname,
                "type": type_text(member.type),
                "payload_sha256": sha256_stream(extracted),
            }
    return records


def inventory_sha256(records: dict[str, dict[str, Any]]) -> str:
    rows = [records[name] for name in sorted(records)]
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_ids_sha256(member_names: list[str]) -> str:
    source_ids = sorted(Path(name).stem for name in member_names)
    return hashlib.sha256(("\n".join(source_ids) + "\n").encode("utf-8")).hexdigest()


def build(parent: Path, excluded_member: str, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite output or partial: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_count = 0
    try:
        with tarfile.open(parent, "r:*") as source, tarfile.open(partial, "w") as target:
            for member in source:
                if member.name == excluded_member:
                    excluded_count += 1
                    continue
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    raise RuntimeError(
                        f"unexpected non-regular/non-WAV member: {member.name}"
                    )
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot read tar member: {member.name}")
                target.addfile(member, extracted)
        if excluded_count != 1:
            raise RuntimeError(
                f"expected exactly one excluded member, observed {excluded_count}"
            )
        partial.replace(output)
    except BaseException:
        # Keep a partial archive as crash evidence; never publish it as the final path.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded-member", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-excluded-payload-sha256", required=True)
    parser.add_argument("--expected-parent-member-count", type=int, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = args.parent.resolve(strict=True)
    output = args.output.absolute()
    audit = args.audit_json.absolute()
    if audit.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {audit}")
    if output.parent.resolve() == parent.parent.resolve() and output.name == parent.name:
        raise RuntimeError("sanitized output must not replace the parent archive")
    free_bytes = shutil.disk_usage(output.parent if output.parent.exists() else output.parent.parent).free
    if free_bytes < parent.stat().st_size + 2 * 1024**3:
        raise RuntimeError("insufficient free space for an independently audited output tar")

    parent_sha256 = sha256_file(parent)
    if parent_sha256 != args.expected_parent_sha256:
        raise RuntimeError("parent archive SHA-256 differs from the approved identity")
    parent_records = inventory(parent)
    if len(parent_records) != args.expected_parent_member_count:
        raise RuntimeError("parent archive member count differs from expectation")
    if args.excluded_member not in parent_records:
        raise RuntimeError("excluded member is absent from parent archive")
    if (
        parent_records[args.excluded_member]["payload_sha256"]
        != args.expected_excluded_payload_sha256
    ):
        raise RuntimeError("excluded member payload differs from diagnostic evidence")

    build(parent, args.excluded_member, output)
    output_sha256 = sha256_file(output)
    output_records = inventory(output)
    expected_records = {
        name: record
        for name, record in parent_records.items()
        if name != args.excluded_member
    }
    missing = sorted(set(expected_records) - set(output_records))
    extra = sorted(set(output_records) - set(expected_records))
    mismatches = [
        {
            "member": name,
            "parent": expected_records[name],
            "sanitized": output_records[name],
        }
        for name in sorted(set(expected_records) & set(output_records))
        if expected_records[name] != output_records[name]
    ]
    expected_identity = inventory_sha256(expected_records)
    output_identity = inventory_sha256(output_records)
    gate_passed = (
        len(output_records) == args.expected_parent_member_count - 1
        and args.excluded_member not in output_records
        and not missing
        and not extra
        and not mismatches
        and expected_identity == output_identity
    )
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "completed_at_utc": utc_now(),
        "parent_official_archive": str(parent),
        "parent_official_archive_bytes": parent.stat().st_size,
        "parent_official_archive_sha256": parent_sha256,
        "sanitized_archive": str(output.resolve()),
        "sanitized_archive_bytes": output.stat().st_size,
        "sanitized_archive_sha256": output_sha256,
        "excluded_member": args.excluded_member,
        "excluded_payload_sha256": parent_records[args.excluded_member]["payload_sha256"],
        "parent_member_count": len(parent_records),
        "sanitized_member_count": len(output_records),
        "excluded_present_in_parent": args.excluded_member in parent_records,
        "excluded_present_in_sanitized": args.excluded_member in output_records,
        "missing_vs_parent_minus_excluded": missing,
        "extra_vs_parent_minus_excluded": extra,
        "member_identity_fields": list(IDENTITY_FIELDS),
        "remaining_member_identity_mismatch_count": len(mismatches),
        "remaining_member_identity_mismatches": mismatches,
        "parent_minus_excluded_identity_sha256": expected_identity,
        "sanitized_identity_sha256": output_identity,
        "sanitized_source_ids_sha256": source_ids_sha256(list(output_records)),
        "gate_passed": gate_passed,
    }
    atomic_json(audit, report)
    if not gate_passed:
        raise RuntimeError(f"sanitized archive audit failed; evidence={audit}")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
