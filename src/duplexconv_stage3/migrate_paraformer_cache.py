"""Migrate accepted ASR results to the collision-safe view cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .paraformer_inference import (
    _validate_cached_result,
    _write_cache_record,
    make_cache_signature,
)
from .state_labeling import read_jsonl


def migrate(*, asr_dir: Path, audio_dir: Path, cache_dir: Path) -> dict:
    asr_dir = asr_dir.resolve(strict=True)
    audio_dir = audio_dir.resolve(strict=True)
    cache_dir = cache_dir.absolute()
    summary = json.loads((asr_dir / "summary.json").read_text(encoding="utf-8"))
    audio_by_view = {
        item["view_id"]: item for item in read_jsonl(audio_dir / "audio_manifest.jsonl")
    }
    migrated = 0
    for result in read_jsonl(asr_dir / "asr_results.jsonl"):
        audio = audio_by_view[result["view_id"]]
        signature = make_cache_signature(
            audio=audio,
            model_sha256=summary["model_key_sha256"],
            funasr_version=summary["funasr_version"],
        )
        updated = {**result, "cache_signature": signature}
        _validate_cached_result(
            {"cache_schema_version": 1, **updated},
            cache_signature=signature,
            view_id=result["view_id"],
            audio_sha256=audio["audio_sha256"],
            model_sha256=summary["model_key_sha256"],
        )
        _write_cache_record(cache_dir / f"{signature}.json", updated)
        migrated += 1
    return {"migrated_view_count": migrated, "cache_dir": str(cache_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            migrate(asr_dir=args.asr_dir, audio_dir=args.audio_dir, cache_dir=args.cache_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
