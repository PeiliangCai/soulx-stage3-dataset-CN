"""Extract fixed GLM-4-Voice audio tokens for Stage 3 chunks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import wave

import numpy as np

from .source_scan import CHUNK_MS, sha256_file
from .state_labeling import canonical_json, read_jsonl


SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
TOKEN_SAMPLES = 1280
AUDIO_TOKEN_COUNT_PER_CHUNK = 2
AUDIO_TOKEN_VOCAB_SIZE = 51866
GLM_AUDIO_PROFILE = "glm4voice-vq-30s-contiguous-80ms-v1"
CACHE_SCHEMA_VERSION = 1
GLM_CACHE_SIGNATURE_PROFILE = "glm-view-cache-v2"


def split_audio_segments(audio: np.ndarray, segment_samples: int = 30 * SAMPLE_RATE) -> list[np.ndarray]:
    if audio.ndim != 1 or audio.dtype != np.float32:
        raise ValueError("audio must be mono float32")
    if not len(audio):
        return []
    return [audio[start : start + segment_samples] for start in range(0, len(audio), segment_samples)]


def group_chunk_tokens(tokens: Sequence[int], chunk_count: int) -> list[list[int]]:
    expected = chunk_count * AUDIO_TOKEN_COUNT_PER_CHUNK
    if len(tokens) != expected:
        raise ValueError(f"GLM token count mismatch: {len(tokens)} != {expected}")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
        raise ValueError("GLM tokens must be integers")
    if any(token < 0 or token >= AUDIO_TOKEN_VOCAB_SIZE for token in tokens):
        raise ValueError("GLM token is outside the official audio vocabulary")
    return [list(tokens[start : start + 2]) for start in range(0, len(tokens), 2)]


def _load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(f"unexpected WAV format: {path}")
        raw = reader.readframes(reader.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _extract_view_tokens(
    *, model: Any, feature_extractor: Any, audio: np.ndarray, device: str, batch_size: int
) -> list[int]:
    import torch

    pooling_kernel_size = model.config.pooling_kernel_size or 1
    convolution_stride = model.conv1.stride[0] * model.conv2.stride[0]
    feature_stride = convolution_stride * pooling_kernel_size * feature_extractor.hop_length
    segments = split_audio_segments(audio)
    all_tokens: list[int] = []
    with torch.no_grad():
        for start in range(0, len(segments), batch_size):
            features = feature_extractor(
                segments[start : start + batch_size],
                sampling_rate=SAMPLE_RATE,
                return_attention_mask=True,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=feature_stride,
            ).to(device=device)
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::convolution_stride]
            attention_mask = attention_mask[:, ::pooling_kernel_size]
            if attention_mask.shape != speech_tokens.shape:
                raise ValueError("GLM attention/token shape mismatch")
            for row in range(len(speech_tokens)):
                all_tokens.extend(speech_tokens[row][attention_mask[row].bool()].tolist())
    return all_tokens


def _validate_cache(
    cached: dict[str, Any], *, signature: str, view_id: str, chunk_count: int
) -> dict[str, Any]:
    if cached.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"GLM cache schema mismatch for {view_id}")
    if cached.get("cache_signature") != signature or cached.get("view_id") != view_id:
        raise ValueError(f"GLM cache identity mismatch for {view_id}")
    if cached.get("effective_chunk_count") != chunk_count:
        raise ValueError(f"GLM cache chunk count mismatch for {view_id}")
    group_chunk_tokens(
        [token for pair in cached.get("chunk_audio_tokens", []) for token in pair],
        chunk_count,
    )
    return {key: value for key, value in cached.items() if key != "cache_schema_version"}


def _write_cache(path: Path, result: dict[str, Any]) -> None:
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


def make_glm_cache_signature(
    *, manifest: dict[str, Any], timeline: dict[str, Any], model_signature: str
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "cache_signature_profile": GLM_CACHE_SIGNATURE_PROFILE,
                "view_id": manifest["view_id"],
                "audio_sha256": manifest["audio_sha256"],
                "terminal_silence_padding_chunks": timeline[
                    "terminal_silence_padding_chunks"
                ],
                "model_signature": model_signature,
                "glm_audio_profile": GLM_AUDIO_PROFILE,
            }
        ).encode("utf-8")
    ).hexdigest()


def extract_glm_audio_tokens(
    *,
    upstream_dir: Path,
    model_dir: Path,
    audio_dir: Path,
    timeline_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 16,
    selected_view_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    upstream_dir = upstream_dir.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    audio_dir = audio_dir.resolve(strict=True)
    timeline_dir = timeline_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    cache_dir = cache_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    temporary.mkdir()
    try:
        import torch
        from transformers import WhisperFeatureExtractor

        sys.path.insert(0, str(upstream_dir))
        from models.glm_4_voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder

        manifests = list(read_jsonl(audio_dir / "audio_manifest.jsonl"))
        timelines = {item["view_id"]: item for item in read_jsonl(timeline_dir / "timelines.jsonl")}
        if selected_view_ids is not None:
            selected = set(selected_view_ids)
            if len(selected) != len(selected_view_ids):
                raise ValueError("selected view IDs contain duplicates")
            manifests = [item for item in manifests if item["view_id"] in selected]
            missing = selected - {item["view_id"] for item in manifests}
            if missing:
                raise ValueError(f"selected view IDs are unknown: {sorted(missing)}")
        manifests.sort(key=lambda item: item["view_id"])

        model_files = [model_dir / "model.safetensors", model_dir / "config.json"]
        model_signature = hashlib.sha256(
            canonical_json({path.name: sha256_file(path) for path in model_files}).encode("utf-8")
        ).hexdigest()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model = None
        feature_extractor = None

        results = []
        quarantines = []
        cache_hits = 0
        cache_misses = 0
        for manifest in manifests:
            view_id = manifest["view_id"]
            timeline = timelines[view_id]
            effective_chunk_count = timeline["effective_chunk_count"]
            signature = make_glm_cache_signature(
                manifest=manifest,
                timeline=timeline,
                model_signature=model_signature,
            )
            cache_path = cache_dir / f"{signature}.json"
            try:
                if cache_path.exists():
                    result = _validate_cache(
                        json.loads(cache_path.read_text(encoding="utf-8")),
                        signature=signature,
                        view_id=view_id,
                        chunk_count=effective_chunk_count,
                    )
                    cache_hits += 1
                else:
                    cache_misses += 1
                    if model is None:
                        model = WhisperVQEncoder.from_pretrained(str(model_dir)).eval().to(device)
                        feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
                    audio = _load_wav_float32(audio_dir / manifest["wav_relative_path"])
                    expected_original_samples = manifest["chunk_count"] * CHUNK_SAMPLES
                    if len(audio) != expected_original_samples:
                        raise ValueError(f"WAV/chunk sample mismatch for {view_id}")
                    padding_chunks = timeline["terminal_silence_padding_chunks"]
                    if padding_chunks:
                        audio = np.pad(audio, (0, padding_chunks * CHUNK_SAMPLES))
                    raw_tokens = _extract_view_tokens(
                        model=model,
                        feature_extractor=feature_extractor,
                        audio=audio,
                        device=device,
                        batch_size=batch_size,
                    )
                    chunk_audio_tokens = group_chunk_tokens(raw_tokens, effective_chunk_count)
                    result = {
                        "view_id": view_id,
                        "source_id": manifest["source_id"],
                        "source_ntrack": manifest["source_ntrack"],
                        "target_channel": manifest["target_channel"],
                        "effective_chunk_count": effective_chunk_count,
                        "terminal_silence_padding_chunks": padding_chunks,
                        "audio_sha256": manifest["audio_sha256"],
                        "glm_model_signature": model_signature,
                        "glm_audio_profile": GLM_AUDIO_PROFILE,
                        "cache_signature": signature,
                        "chunk_audio_tokens": chunk_audio_tokens,
                    }
                    _write_cache(cache_path, result)
                results.append(result)
            except Exception as exc:
                quarantines.append(
                    {
                        "view_id": view_id,
                        "source_id": manifest["source_id"],
                        "cache_signature": signature,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

        with (temporary / "audio_tokens.jsonl").open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(canonical_json(item) + "\n")
        with (temporary / "glm_quarantine.jsonl").open("w", encoding="utf-8") as handle:
            for item in quarantines:
                handle.write(canonical_json(item) + "\n")
        summary = {
            "schema_version": 1,
            "glm_audio_profile": GLM_AUDIO_PROFILE,
            "model_dir": str(model_dir),
            "model_signature": model_signature,
            "input_view_count": len(manifests),
            "passed_view_count": len(results),
            "quarantined_view_count": len(quarantines),
            "cache_hit_count": cache_hits,
            "cache_miss_count": cache_misses,
            "effective_chunk_count": sum(item["effective_chunk_count"] for item in results),
            "audio_token_count": sum(
                2 * item["effective_chunk_count"] for item in results
            ),
            "view_count_by_ntrack": dict(
                sorted(Counter(str(item["source_ntrack"]) for item in results).items())
            ),
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
        (temporary / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract fixed GLM-4-Voice audio tokens.")
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--view-id", action="append")
    parser.add_argument("--all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all == bool(args.view_id):
        raise ValueError("choose exactly one of --all or one/more --view-id")
    summary = extract_glm_audio_tokens(
        upstream_dir=args.upstream_dir,
        model_dir=args.model_dir,
        audio_dir=args.audio_dir,
        timeline_dir=args.timeline_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        selected_view_ids=None if args.all else args.view_id,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
