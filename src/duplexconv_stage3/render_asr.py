"""Deterministically render mixed Chinese/ASCII Paraformer tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Sequence

from .source_scan import sha256_file
from .state_labeling import canonical_json, read_jsonl


TEXT_RENDER_PROFILE = "mixed-zh-ascii-token-spacing-v1"
_ASCII_WORD = re.compile(r"[A-Za-z0-9]")


def render_token_group(tokens: Sequence[str]) -> str:
    rendered = ""
    previous_was_ascii = False
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("ASR token must be a non-empty string")
        is_ascii = bool(_ASCII_WORD.search(token)) and not bool(
            re.search(r"[\u4e00-\u9fff]", token)
        )
        if rendered and previous_was_ascii:
            rendered += " "
        rendered += token
        previous_was_ascii = is_ascii
    return rendered


def render_asr_result(item: dict[str, Any]) -> dict[str, Any]:
    chunk_count = item["chunk_count"]
    grouped: list[list[str]] = [[] for _ in range(chunk_count)]
    for expected_index, token in enumerate(item["tokens"]):
        if token["index"] != expected_index:
            raise ValueError(f"non-contiguous token index for {item['view_id']}")
        emit_chunk = token["emit_chunk"]
        if not 0 <= emit_chunk < chunk_count:
            raise ValueError(f"token emit chunk out of bounds for {item['view_id']}")
        grouped[emit_chunk].append(token["token"])
    chunk_targets = [render_token_group(tokens) for tokens in grouped]
    rendered_text = "".join(chunk_targets)
    render_signature = hashlib.sha256(
        canonical_json(
            {
                "asr_cache_signature": item["cache_signature"],
                "text_render_profile": TEXT_RENDER_PROFILE,
                "chunk_asr_targets": chunk_targets,
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        **item,
        "text": rendered_text,
        "chunk_asr_targets": chunk_targets,
        "text_render_profile": TEXT_RENDER_PROFILE,
        "render_signature": render_signature,
    }


def validate_asr_partition(
    results: Sequence[dict[str, Any]], quarantines: Sequence[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    result_ids = [item["view_id"] for item in results]
    quarantine_ids = [item["view_id"] for item in quarantines]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("input ASR results contain duplicate view IDs")
    if len(set(quarantine_ids)) != len(quarantine_ids):
        raise ValueError("input ASR quarantine contains duplicate view IDs")
    overlap = set(result_ids) & set(quarantine_ids)
    if overlap:
        raise ValueError(
            f"ASR result/quarantine view IDs overlap: {sorted(overlap)}"
        )
    return set(result_ids), set(quarantine_ids)


def render_asr_directory(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    temporary.mkdir()
    try:
        quarantines = list(read_jsonl(input_dir / "asr_quarantine.jsonl"))
        input_results = list(read_jsonl(input_dir / "asr_results.jsonl"))
        result_ids, quarantine_ids = validate_asr_partition(input_results, quarantines)
        input_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
        input_view_count = input_summary.get("input_view_count")
        if input_view_count != len(result_ids) + len(quarantine_ids):
            raise ValueError(
                "input ASR summary does not close over results and quarantine"
            )
        if input_summary.get("passed_view_count") != len(result_ids):
            raise ValueError("input ASR summary passed-view count mismatch")
        if input_summary.get("quarantined_view_count") != len(quarantine_ids):
            raise ValueError("input ASR summary quarantine count mismatch")
        results = [render_asr_result(item) for item in input_results]
        with (temporary / "asr_results.jsonl").open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(canonical_json(item) + "\n")
        with (temporary / "asr_quarantine.jsonl").open("w", encoding="utf-8") as handle:
            for item in quarantines:
                handle.write(canonical_json(item) + "\n")
        summary = {
            **input_summary,
            "rendered_from": str(input_dir),
            "text_render_profile": TEXT_RENDER_PROFILE,
            "rendered_view_count": len(results),
            "propagated_quarantined_view_count": len(quarantines),
            "view_partition_closed": True,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (temporary / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render mixed Chinese/ASCII ASR text.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = render_asr_directory(input_dir=args.input_dir, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
