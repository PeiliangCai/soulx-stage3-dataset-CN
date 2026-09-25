#!/usr/bin/env python3
"""Build a deterministic Gate D tar from the views that reach model-ready."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready-dir", type=Path, required=True)
    parser.add_argument("--target-audio-dir", type=Path, required=True)
    parser.add_argument("--selection-list", type=Path, required=True)
    parser.add_argument("--output-tar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for output in (args.selection_list, args.output_tar, args.manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    windows_path = args.model_ready_dir / "metadata/windows.jsonl"
    audio_manifest_path = args.target_audio_dir / "audio_manifest.jsonl"
    windows = read_jsonl(windows_path)
    audio_manifest = read_jsonl(audio_manifest_path)
    view_ids = sorted({record["view_id"] for record in windows})
    audio_by_view = {
        record["view_id"]: record["wav_relative_path"] for record in audio_manifest
    }
    missing = sorted(set(view_ids) - set(audio_by_view))
    if missing:
        raise RuntimeError(f"model-ready views are absent from audio manifest: {missing}")
    members = sorted(audio_by_view[view_id] for view_id in view_ids)
    if len(members) != len(set(members)):
        raise RuntimeError("selected audio members are not unique")
    for member in members:
        if not (args.target_audio_dir / member).is_file():
            raise FileNotFoundError(args.target_audio_dir / member)

    temporary_list = args.selection_list.with_name(
        f".{args.selection_list.name}.tmp-{os.getpid()}"
    )
    temporary_list.write_text("".join(f"{member}\n" for member in members), encoding="utf-8")
    temporary_list.replace(args.selection_list)

    temporary_tar = args.output_tar.with_name(f".{args.output_tar.name}.tmp-{os.getpid()}")
    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--format=ustar",
            "--no-recursion",
            "-cf",
            str(temporary_tar),
            "-C",
            str(args.target_audio_dir),
            "-T",
            str(args.selection_list),
        ],
        check=True,
    )
    temporary_tar.replace(args.output_tar)
    with tarfile.open(args.output_tar, "r") as archive:
        tar_members = [member.name for member in archive.getmembers() if member.isfile()]
    if tar_members != members:
        raise RuntimeError("deterministic tar members do not match the selection list")

    manifest = {
        "schema_version": 1,
        "profile": "duplexconv-model-ready-gate-d-input-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_definition": "All unique view_id values contributing at least one model-ready window",
        "model_ready_view_count": len(view_ids),
        "source_id_count": len({record["source_id"] for record in windows}),
        "selection_member_count": len(members),
        "tar_member_count": len(tar_members),
        "partition_closed": True,
        "selection_list": str(args.selection_list),
        "selection_list_sha256": sha256(args.selection_list),
        "output_tar": str(args.output_tar),
        "output_tar_bytes": args.output_tar.stat().st_size,
        "output_tar_sha256": sha256(args.output_tar),
        "model_ready_windows_sha256": sha256(windows_path),
        "target_audio_manifest_sha256": sha256(audio_manifest_path),
    }
    temporary_manifest = args.manifest.with_name(f".{args.manifest.name}.tmp-{os.getpid()}")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
