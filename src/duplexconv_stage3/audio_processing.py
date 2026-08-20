"""Deterministic target-channel extraction for DuplexConv Stage 3."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Iterable, Sequence
import wave

import numpy as np
from scipy.signal import resample_poly

from .source_scan import CHUNK_MS, sha256_file
from .state_labeling import canonical_json, read_jsonl


TARGET_SAMPLE_RATE = 16000
SOURCE_SAMPLE_RATE = 48000
RESAMPLE_PROFILE = "scipy-resample-poly-1to3-kaiser5-s16-v1"
AUDIO_PROFILE = "target-mono-16k-s16-chunkpad-v1"


def safe_view_filename(view_id: str) -> str:
    return view_id.replace("/", "__") + ".wav"


def resample_pcm16_48k_to_16k(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError("input must be a one-dimensional int16 array")
    float_samples = samples.astype(np.float32) / 32768.0
    resampled = resample_poly(
        float_samples,
        up=1,
        down=3,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    quantized = np.rint(resampled * 32768.0)
    return np.clip(quantized, -32768, 32767).astype("<i2")


def pad_to_chunk_boundary(
    samples: np.ndarray, chunk_samples: int = TARGET_SAMPLE_RATE * CHUNK_MS // 1000
) -> tuple[np.ndarray, int]:
    if samples.ndim != 1:
        raise ValueError("samples must be one dimensional")
    padding = (-len(samples)) % chunk_samples
    if padding:
        samples = np.pad(samples, (0, padding), mode="constant")
    return samples, padding


def _write_pcm16_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TARGET_SAMPLE_RATE)
        writer.writeframes(samples.astype("<i2", copy=False).tobytes())


def extract_target_audio(
    *,
    audio_archive: Path,
    scan_dir: Path,
    output_dir: Path,
    selected_view_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    audio_archive = audio_archive.resolve(strict=True)
    scan_dir = scan_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    temporary.mkdir()
    audio_output = temporary / "audio"
    audio_output.mkdir()
    try:
        views = list(read_jsonl(scan_dir / "target_views.jsonl"))
        source_inventory = {
            item["source_id"]: item
            for item in read_jsonl(scan_dir / "source_inventory.jsonl")
        }
        if selected_view_ids is not None:
            selected = set(selected_view_ids)
            if len(selected) != len(selected_view_ids):
                raise ValueError("selected view IDs contain duplicates")
            views = [view for view in views if view["view_id"] in selected]
            missing = selected - {view["view_id"] for view in views}
            if missing:
                raise ValueError(f"selected view IDs are unknown: {sorted(missing)}")
        views.sort(key=lambda item: item["view_id"])
        views_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for view in views:
            views_by_source[view["source_id"]].append(view)

        scan_summary = json.loads(
            (scan_dir / "summary.json").read_text(encoding="utf-8")
        )
        archive_sha256 = scan_summary["inputs"]["audio_archive_sha256"]
        manifest: list[dict[str, Any]] = []
        with tarfile.open(audio_archive, "r:*") as audio_tar:
            members = {
                Path(member.name).stem: member
                for member in audio_tar.getmembers()
                if member.isfile() and member.name.endswith(".wav")
            }
            for source_id, source_views in sorted(views_by_source.items()):
                source = source_inventory[source_id]
                member = members[source_id]
                extracted = audio_tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"cannot extract source WAV: {source_id}")
                with wave.open(extracted, "rb") as reader:
                    if (
                        reader.getnchannels() != source["ntrack"]
                        or reader.getframerate() != SOURCE_SAMPLE_RATE
                        or reader.getsampwidth() != 2
                        or reader.getnframes() != source["frame_count"]
                    ):
                        raise RuntimeError(f"WAV contract changed for {source_id}")
                    raw = reader.readframes(reader.getnframes())
                interleaved = np.frombuffer(raw, dtype="<i2")
                expected_samples = source["frame_count"] * source["ntrack"]
                if interleaved.size != expected_samples:
                    raise RuntimeError(f"PCM sample count mismatch for {source_id}")
                channels = interleaved.reshape(-1, source["ntrack"])

                for view in sorted(source_views, key=lambda item: item["target_channel"]):
                    channel = channels[:, view["target_channel"]].copy()
                    resampled = resample_pcm16_48k_to_16k(channel)
                    expected_resampled = math.ceil(len(channel) / 3)
                    if len(resampled) != expected_resampled:
                        raise RuntimeError(
                            f"resampled length mismatch for {view['view_id']}: "
                            f"{len(resampled)} != {expected_resampled}"
                        )
                    padded, padding_samples = pad_to_chunk_boundary(resampled)
                    if len(padded) // (TARGET_SAMPLE_RATE * CHUNK_MS // 1000) != view["chunk_count"]:
                        raise RuntimeError(f"chunk count mismatch for {view['view_id']}")
                    filename = safe_view_filename(view["view_id"])
                    destination = audio_output / filename
                    _write_pcm16_wav(destination, padded)
                    audio_sha256 = sha256_file(destination)
                    signature = hashlib.sha256(
                        canonical_json(
                            {
                                "source_archive_sha256": archive_sha256,
                                "source_id": source_id,
                                "source_frame_count": source["frame_count"],
                                "target_channel": view["target_channel"],
                                "resample_profile": RESAMPLE_PROFILE,
                                "audio_profile": AUDIO_PROFILE,
                                "chunk_ms": CHUNK_MS,
                            }
                        ).encode("utf-8")
                    ).hexdigest()
                    manifest.append(
                        {
                            "view_id": view["view_id"],
                            "source_id": source_id,
                            "source_ntrack": view["source_ntrack"],
                            "conversation_domain": view["conversation_domain"],
                            "target_channel": view["target_channel"],
                            "reference_channels": view["reference_channels"],
                            "wav_relative_path": f"audio/{filename}",
                            "sample_rate": TARGET_SAMPLE_RATE,
                            "sample_width_bytes": 2,
                            "channel_count": 1,
                            "source_frame_count": source["frame_count"],
                            "resampled_frame_count": len(resampled),
                            "padding_samples": padding_samples,
                            "padded_frame_count": len(padded),
                            "chunk_count": view["chunk_count"],
                            "duration_ms": 1000.0 * len(resampled) / TARGET_SAMPLE_RATE,
                            "padded_duration_ms": 1000.0 * len(padded) / TARGET_SAMPLE_RATE,
                            "target_active_chunks": sum(
                                view["target_active_by_chunk"]
                            ),
                            "audio_sha256": audio_sha256,
                            "audio_cache_signature": signature,
                            "resample_profile": RESAMPLE_PROFILE,
                            "audio_profile": AUDIO_PROFILE,
                        }
                    )
        manifest_path = temporary / "audio_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for item in manifest:
                handle.write(canonical_json(item) + "\n")
        summary = {
            "schema_version": 1,
            "source_archive": str(audio_archive),
            "source_archive_sha256": archive_sha256,
            "scan_dir": str(scan_dir),
            "view_count": len(manifest),
            "view_count_by_ntrack": {
                str(ntrack): sum(item["source_ntrack"] == ntrack for item in manifest)
                for ntrack in (2, 3)
            },
            "resample_profile": RESAMPLE_PROFILE,
            "audio_profile": AUDIO_PROFILE,
            "total_resampled_frames": sum(
                item["resampled_frame_count"] for item in manifest
            ),
            "total_padding_samples": sum(item["padding_samples"] for item in manifest),
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                checksum_lines.append(
                    f"{sha256_file(path)}  {path.relative_to(temporary)}\n"
                )
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
    parser = argparse.ArgumentParser(description="Extract deterministic target audio.")
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-id", action="append")
    parser.add_argument("--all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all == bool(args.view_id):
        raise ValueError("choose exactly one of --all or one/more --view-id")
    summary = extract_target_audio(
        audio_archive=args.audio_archive,
        scan_dir=args.scan_dir,
        output_dir=args.output_dir,
        selected_view_ids=None if args.all else args.view_id,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
