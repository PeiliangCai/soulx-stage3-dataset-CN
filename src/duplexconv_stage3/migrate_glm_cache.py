"""Migrate accepted GLM results to the collision-safe view cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .glm_audio_tokens import (
    _validate_cache,
    _write_cache,
    make_glm_cache_signature,
)
from .state_labeling import read_jsonl


def migrate(
    *, glm_dir: Path, audio_dir: Path, timeline_dir: Path, cache_dir: Path
) -> dict:
    glm_dir = glm_dir.resolve(strict=True)
    audio_dir = audio_dir.resolve(strict=True)
    timeline_dir = timeline_dir.resolve(strict=True)
    cache_dir = cache_dir.absolute()
    summary = json.loads((glm_dir / "summary.json").read_text(encoding="utf-8"))
    audio_by_view = {
        item["view_id"]: item for item in read_jsonl(audio_dir / "audio_manifest.jsonl")
    }
    timeline_by_view = {
        item["view_id"]: item for item in read_jsonl(timeline_dir / "timelines.jsonl")
    }
    migrated = 0
    for result in read_jsonl(glm_dir / "audio_tokens.jsonl"):
        view_id = result["view_id"]
        signature = make_glm_cache_signature(
            manifest=audio_by_view[view_id],
            timeline=timeline_by_view[view_id],
            model_signature=summary["model_signature"],
        )
        updated = {**result, "cache_signature": signature}
        _validate_cache(
            {"cache_schema_version": 1, **updated},
            signature=signature,
            view_id=view_id,
            chunk_count=timeline_by_view[view_id]["effective_chunk_count"],
        )
        _write_cache(cache_dir / f"{signature}.json", updated)
        migrated += 1
    return {"migrated_view_count": migrated, "cache_dir": str(cache_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            migrate(
                glm_dir=args.glm_dir,
                audio_dir=args.audio_dir,
                timeline_dir=args.timeline_dir,
                cache_dir=args.cache_dir,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
