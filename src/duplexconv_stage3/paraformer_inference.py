"""Run fixed local Paraformer and map each token to one Stage 3 chunk."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

from .source_scan import CHUNK_MS, interval_chunk_bounds, sha256_file
from .state_labeling import canonical_json, read_jsonl


ASR_PROFILE = "paraformer-pseudolabel-v1"
TOKEN_EMIT_PROFILE = "token-end-ceil-160ms-v1"
ACTIVITY_EVIDENCE_PROFILE = "token-interval-overlap-320ms-v2"
ACTIVITY_TOLERANCE_CHUNKS = 2
CACHE_SCHEMA_VERSION = 1
CACHE_SIGNATURE_PROFILE = "paraformer-view-cache-v2"


def parse_paraformer_result(
    result: dict[str, Any],
    *,
    chunk_count: int,
    source_duration_ms: float,
) -> dict[str, Any]:
    text = result.get("text")
    timestamps = result.get("timestamp")
    if not isinstance(text, str):
        raise ValueError("Paraformer text is not a string")
    if not isinstance(timestamps, list):
        raise ValueError("Paraformer timestamp is not a list")
    tokens = text.split()
    if len(tokens) != len(timestamps):
        raise ValueError(
            f"token/timestamp count mismatch: {len(tokens)} != {len(timestamps)}"
        )
    if any("<|" in token or "|>" in token for token in tokens):
        raise ValueError("Paraformer output contains a SoulX control token")
    token_records = []
    previous_start = -1.0
    previous_end = -1.0
    chunk_tokens: list[list[str]] = [[] for _ in range(chunk_count)]
    for index, (token, timestamp) in enumerate(zip(tokens, timestamps)):
        if (
            not isinstance(timestamp, (list, tuple))
            or len(timestamp) != 2
            or isinstance(timestamp[0], bool)
            or isinstance(timestamp[1], bool)
            or not isinstance(timestamp[0], (int, float))
            or not isinstance(timestamp[1], (int, float))
        ):
            raise ValueError(f"invalid timestamp at token {index}")
        start_ms = float(timestamp[0])
        end_ms = float(timestamp[1])
        if (
            not math.isfinite(start_ms)
            or not math.isfinite(end_ms)
            or start_ms < 0
            or end_ms < start_ms
            or start_ms < previous_start
            or end_ms < previous_end
            or end_ms > source_duration_ms + 1.0
        ):
            raise ValueError(
                f"non-monotonic or out-of-bounds timestamp at token {index}: "
                f"start={start_ms}, end={end_ms}, previous_start={previous_start}, "
                f"previous_end={previous_end}, audio_end={source_duration_ms}"
            )
        emit_chunk = max(0, min(chunk_count - 1, math.ceil(end_ms / CHUNK_MS) - 1))
        chunk_tokens[emit_chunk].append(token)
        token_records.append(
            {
                "index": index,
                "token": token,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "emit_chunk": emit_chunk,
            }
        )
        previous_start = start_ms
        previous_end = end_ms
    return {
        "text_with_token_spaces": text,
        "text": "".join(tokens),
        "tokens": token_records,
        "chunk_asr_targets": ["".join(items) for items in chunk_tokens],
    }


def _token_has_activity_evidence(
    token: dict[str, Any],
    view: dict[str, Any],
    fallback_events: Sequence[dict[str, Any]],
) -> tuple[bool, list[str]]:
    # Paraformer occasionally stretches the final character over a following
    # silence interval.  Keep end-based emission unchanged for Stage 3, but
    # judge acoustic evidence by whether the token's full timestamp interval
    # overlaps target activity (with the documented 320 ms tolerance).
    token_start, token_stop = interval_chunk_bounds(
        token["start_ms"], token["end_ms"], view["chunk_count"]
    )
    start = max(0, token_start - ACTIVITY_TOLERANCE_CHUNKS)
    stop = min(view["chunk_count"], token_stop + ACTIVITY_TOLERANCE_CHUNKS)
    has_primary_activity = any(view["target_active_by_chunk"][start:stop])
    matching_fallback_ids = []
    for event in fallback_events:
        event_start, event_end = event["effective_event_envelope_ms"]
        event_chunk_start, event_chunk_stop = interval_chunk_bounds(
            event_start, event_end, view["chunk_count"]
        )
        if (
            token_start < event_chunk_stop + ACTIVITY_TOLERANCE_CHUNKS
            and token_stop > event_chunk_start - ACTIVITY_TOLERANCE_CHUNKS
        ):
            matching_fallback_ids.append(event["event_id"])
    return has_primary_activity or bool(matching_fallback_ids), matching_fallback_ids


def _validate_cached_result(
    cached: dict[str, Any],
    *,
    cache_signature: str,
    view_id: str,
    audio_sha256: str,
    model_sha256: str,
) -> dict[str, Any]:
    expected = {
        "cache_signature": cache_signature,
        "view_id": view_id,
        "audio_sha256": audio_sha256,
        "asr_model_key_sha256": model_sha256,
        "asr_supervision_profile": ASR_PROFILE,
        "token_emit_profile": TOKEN_EMIT_PROFILE,
        "activity_evidence_profile": ACTIVITY_EVIDENCE_PROFILE,
    }
    if cached.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"cache schema mismatch for {view_id}")
    for key, value in expected.items():
        if cached.get(key) != value:
            raise ValueError(f"cache {key} mismatch for {view_id}")
    if not isinstance(cached.get("tokens"), list) or not isinstance(
        cached.get("chunk_asr_targets"), list
    ):
        raise ValueError(f"cache payload is incomplete for {view_id}")
    if len(cached["chunk_asr_targets"]) != cached.get("chunk_count"):
        raise ValueError(f"cache chunk count mismatch for {view_id}")
    return {key: value for key, value in cached.items() if key != "cache_schema_version"}


def _write_cache_record(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    temporary.write_text(
        canonical_json({"cache_schema_version": CACHE_SCHEMA_VERSION, **result}) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def make_cache_signature(
    *, audio: dict[str, Any], model_sha256: str, funasr_version: str
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "cache_signature_profile": CACHE_SIGNATURE_PROFILE,
                "view_id": audio["view_id"],
                "audio_sha256": audio["audio_sha256"],
                "model_sha256": model_sha256,
                "funasr_version": funasr_version,
                "asr_profile": ASR_PROFILE,
                "token_emit_profile": TOKEN_EMIT_PROFILE,
                "activity_evidence_profile": ACTIVITY_EVIDENCE_PROFILE,
            }
        ).encode("utf-8")
    ).hexdigest()


def run_paraformer(
    *,
    model_dir: Path,
    audio_dir: Path,
    scan_dir: Path,
    output_dir: Path,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    selected_view_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.resolve(strict=True)
    audio_dir = audio_dir.resolve(strict=True)
    scan_dir = scan_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    cache_dir = cache_dir.absolute() if cache_dir is not None else None
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    temporary.mkdir()
    try:
        from funasr import AutoModel, __version__ as funasr_version
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started_at = time.monotonic()

        manifest = list(read_jsonl(audio_dir / "audio_manifest.jsonl"))
        if selected_view_ids is not None:
            selected = set(selected_view_ids)
            if len(selected) != len(selected_view_ids):
                raise ValueError("selected view IDs contain duplicates")
            manifest = [item for item in manifest if item["view_id"] in selected]
            missing = selected - {item["view_id"] for item in manifest}
            if missing:
                raise ValueError(f"selected view IDs are unknown: {sorted(missing)}")
        views = {
            item["view_id"]: item
            for item in read_jsonl(scan_dir / "target_views.jsonl")
        }
        fallback_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in read_jsonl(scan_dir / "events.jsonl"):
            if event["activity_fallback_candidate"]:
                view_id = f"{event['source_id']}/target-ch{event['channel']:02d}"
                fallback_by_view[view_id].append(event)
        model_sha256 = sha256_file(model_dir / "model.pt")
        model = None
        results = []
        quarantines = []
        cache_hit_count = 0
        cache_miss_count = 0
        for audio in manifest:
            view_id = audio["view_id"]
            wav_path = audio_dir / audio["wav_relative_path"]
            view = views[view_id]
            cache_signature = make_cache_signature(
                audio=audio,
                model_sha256=model_sha256,
                funasr_version=funasr_version,
            )
            try:
                cache_path = (
                    cache_dir / f"{cache_signature}.json"
                    if cache_dir is not None
                    else None
                )
                if cache_path is not None and cache_path.exists():
                    result = _validate_cached_result(
                        json.loads(cache_path.read_text(encoding="utf-8")),
                        cache_signature=cache_signature,
                        view_id=view_id,
                        audio_sha256=audio["audio_sha256"],
                        model_sha256=model_sha256,
                    )
                    cache_hit_count += 1
                else:
                    cache_miss_count += 1
                    if model is None:
                        model = AutoModel(
                            model=str(model_dir),
                            vad_model=None,
                            punc_model=None,
                            lm_model=None,
                            disable_update=True,
                            device=device,
                        )
                    generated = model.generate(
                        input=str(wav_path), batch_size_s=300, use_itn=False
                    )
                    if not isinstance(generated, list) or len(generated) != 1:
                        raise ValueError("Paraformer did not return exactly one result")
                    parsed = parse_paraformer_result(
                        generated[0],
                        chunk_count=audio["chunk_count"],
                        source_duration_ms=audio["padded_duration_ms"],
                    )
                    if audio["target_active_chunks"] > 0 and not parsed["tokens"]:
                        raise ValueError("semantic target activity returned no text/timestamps")
                    outside = []
                    fallback_ids = set()
                    for token in parsed["tokens"]:
                        accepted, matched_fallback = _token_has_activity_evidence(
                            token, view, fallback_by_view.get(view_id, [])
                        )
                        fallback_ids.update(matched_fallback)
                        if not accepted:
                            outside.append(token["index"])
                    result = {
                        "view_id": view_id,
                        "source_id": audio["source_id"],
                        "source_ntrack": audio["source_ntrack"],
                        "target_channel": audio["target_channel"],
                        "chunk_count": audio["chunk_count"],
                        "audio_sha256": audio["audio_sha256"],
                        "asr_model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        "asr_model_key_sha256": model_sha256,
                        "asr_optional_vad_punc_lm": "disabled",
                        "asr_supervision_profile": ASR_PROFILE,
                        "token_emit_profile": TOKEN_EMIT_PROFILE,
                        "activity_evidence_profile": ACTIVITY_EVIDENCE_PROFILE,
                        "funasr_version": funasr_version,
                        "cache_signature": cache_signature,
                        "fallback_event_ids_with_token_evidence": sorted(fallback_ids),
                        "asr_token_outside_target_activity": outside,
                        **parsed,
                    }
                    if cache_path is not None:
                        _write_cache_record(cache_path, result)
                results.append(result)
            except Exception as exc:
                quarantines.append(
                    {
                        "view_id": view_id,
                        "source_id": audio["source_id"],
                        "reason": f"{type(exc).__name__}: {exc}",
                        "cache_signature": cache_signature,
                    }
                )

        with (temporary / "asr_results.jsonl").open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(canonical_json(item) + "\n")
        with (temporary / "asr_quarantine.jsonl").open("w", encoding="utf-8") as handle:
            for item in quarantines:
                handle.write(canonical_json(item) + "\n")
        summary = {
            "schema_version": 1,
            "model_dir": str(model_dir),
            "model_key_sha256": model_sha256,
            "funasr_version": funasr_version,
            "input_view_count": len(manifest),
            "passed_view_count": len(results),
            "quarantined_view_count": len(quarantines),
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "view_count_by_ntrack": {
                str(ntrack): sum(item["source_ntrack"] == ntrack for item in results)
                for ntrack in (2, 3)
            },
            "total_token_count": sum(len(item["tokens"]) for item in results),
            "outside_activity_token_count": sum(
                len(item["asr_token_outside_target_activity"]) for item in results
            ),
            "fallback_event_count_with_token_evidence": len(
                {
                    event_id
                    for item in results
                    for event_id in item["fallback_event_ids_with_token_evidence"]
                }
            ),
            "asr_profile": ASR_PROFILE,
            "token_emit_profile": TOKEN_EMIT_PROFILE,
            "activity_evidence_profile": ACTIVITY_EVIDENCE_PROFILE,
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
            "cuda_peak_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (temporary / "checksums.sha256").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed local Paraformer inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--view-id", action="append")
    parser.add_argument("--all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all and args.view_id:
        raise ValueError("--all and --view-id are mutually exclusive")
    summary = run_paraformer(
        model_dir=args.model_dir,
        audio_dir=args.audio_dir,
        scan_dir=args.scan_dir,
        output_dir=args.output_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        selected_view_ids=args.view_id,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
