"""Auditable Easy Turn benchmark runner for the official SoulX inference path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import soxr


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 2_560
TERMINAL_STATES = {
    "<|user_complete|>": "complete",
    "<|user_incomplete|>": "incomplete",
}


@dataclass(frozen=True)
class EasyTurnSample:
    sample_id: str
    language: str
    label: str
    wav_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def discover_samples(root: Path, language: str) -> list[EasyTurnSample]:
    root = root.resolve(strict=True)
    if language not in {"en", "zh"}:
        raise ValueError(f"unsupported language: {language}")
    samples: list[EasyTurnSample] = []
    for label in ("complete", "incomplete"):
        label_root = root / label
        if not label_root.is_dir():
            raise FileNotFoundError(label_root)
        for wav_path in sorted(label_root.rglob("*.wav")):
            relative = wav_path.relative_to(root).as_posix()
            samples.append(
                EasyTurnSample(
                    sample_id=f"{language}:{relative}",
                    language=language,
                    label=label,
                    wav_path=wav_path,
                )
            )
    expected = {"en": {"complete": 318, "incomplete": 299}, "zh": {"complete": 300, "incomplete": 300}}
    counts = {label: sum(sample.label == label for sample in samples) for label in expected[language]}
    if counts != expected[language]:
        raise ValueError(f"unexpected Easy Turn inventory for {language}: {counts}")
    return samples


def select_sample_subset(
    samples: Sequence[EasyTurnSample],
    limit: int | None = None,
    limit_per_label: int | None = None,
) -> list[EasyTurnSample]:
    if limit is not None and limit_per_label is not None:
        raise ValueError("--limit and --limit-per-label are mutually exclusive")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        return list(samples[:limit])
    if limit_per_label is not None:
        if limit_per_label <= 0:
            raise ValueError("--limit-per-label must be positive")
        return [
            sample
            for label in ("complete", "incomplete")
            for sample in [item for item in samples if item.label == label][
                :limit_per_label
            ]
        ]
    return list(samples)


def load_audio_16k_mono(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    source_frames, source_channels = audio.shape
    mono = audio.mean(axis=1, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE, quality="HQ")
    mono = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    return mono, {
        "source_sample_rate": sample_rate,
        "source_channels": source_channels,
        "source_frames": source_frames,
        "normalized_frames": int(mono.shape[0]),
        "duration_seconds": float(source_frames / sample_rate),
    }


def stream_chunks(audio: np.ndarray, tail_silence_ms: int) -> Iterable[tuple[np.ndarray, bool]]:
    if audio.ndim != 1 or audio.dtype != np.float32:
        raise ValueError("audio must be mono float32")
    if tail_silence_ms < 0 or tail_silence_ms % 160 != 0:
        raise ValueError("tail_silence_ms must be a non-negative multiple of 160")
    audio_chunks = math.ceil(len(audio) / CHUNK_SAMPLES)
    padded = np.pad(audio, (0, audio_chunks * CHUNK_SAMPLES - len(audio)))
    for offset in range(0, len(padded), CHUNK_SAMPLES):
        yield padded[offset : offset + CHUNK_SAMPLES].astype(np.float32, copy=False), False
    for _ in range(tail_silence_ms // 160):
        yield np.zeros(CHUNK_SAMPLES, dtype=np.float32), True


def attach_raw_state_capture(turn_model: Any) -> None:
    original_infer = turn_model.infer

    def captured_infer(*args: Any, **kwargs: Any):
        result = original_infer(*args, **kwargs)
        turn_model._benchmark_last_raw_state = result[0]
        return result

    turn_model.infer = captured_infer
    turn_model._benchmark_last_raw_state = None


def evaluate_sample(turn_model: Any, sample: EasyTurnSample, tail_silence_ms: int) -> dict[str, Any]:
    audio, audio_info = load_audio_16k_mono(sample.wav_path)
    turn_model.reset()
    turn_model._benchmark_last_raw_state = None
    trace = []
    saw_nonidle = False
    prediction = "no_decision"
    decision_chunk = None
    started = time.perf_counter()
    for chunk_index, (chunk, is_tail_silence) in enumerate(stream_chunks(audio, tail_silence_ms)):
        chunk_started = time.perf_counter()
        response = turn_model.process(chunk)
        elapsed = time.perf_counter() - chunk_started
        raw_state = turn_model._benchmark_last_raw_state
        if raw_state == "<|user_nonidle|>" or response.get("state") == "nonidle":
            saw_nonidle = True
        trace.append(
            {
                "chunk_index": chunk_index,
                "is_tail_silence": is_tail_silence,
                "raw_state": raw_state,
                "api_state": response.get("state"),
                "asr_segment": response.get("asr_segment", ""),
                "elapsed_seconds": elapsed,
            }
        )
        if saw_nonidle and raw_state in TERMINAL_STATES:
            prediction = TERMINAL_STATES[raw_state]
            decision_chunk = chunk_index
            break
    total_elapsed = time.perf_counter() - started
    return {
        **asdict(sample),
        "wav_path": str(sample.wav_path),
        "wav_sha256": sha256_file(sample.wav_path),
        "prediction": prediction,
        "correct": prediction == sample.label,
        "saw_nonidle": saw_nonidle,
        "decision_chunk": decision_chunk,
        "tail_silence_ms": tail_silence_ms,
        "total_elapsed_seconds": total_elapsed,
        "audio": audio_info,
        "trace": trace,
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(records), "by_label": {}}
    for label in ("complete", "incomplete"):
        subset = [record for record in records if record["label"] == label]
        correct = sum(bool(record["correct"]) for record in subset)
        no_decision = sum(record["prediction"] == "no_decision" for record in subset)
        summary["by_label"][label] = {
            "count": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
            "no_decision": no_decision,
        }
    accuracies = [summary["by_label"][label]["accuracy"] for label in ("complete", "incomplete")]
    summary["macro_accuracy"] = sum(accuracies) / len(accuracies) if all(x is not None for x in accuracies) else None
    summary["total_elapsed_seconds"] = sum(record["total_elapsed_seconds"] for record in records)
    return summary


def progress_path_for(output: Path) -> Path:
    return output.with_name(f".{output.name}.progress.jsonl")


def load_or_create_progress(
    path: Path, run_metadata: dict[str, Any], resume: bool
) -> tuple[str, list[dict[str, Any]]]:
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"unfinished progress exists; pass --resume after verifying it: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        if not lines:
            raise ValueError(f"empty progress journal: {path}")
        header = json.loads(lines[0])
        if header.get("kind") != "header" or header.get("schema_version") != 1:
            raise ValueError(f"invalid progress header: {path}")
        if header.get("run") != run_metadata:
            raise ValueError("progress journal does not match the requested benchmark")
        records = []
        seen = set()
        for line_number, line in enumerate(lines[1:], start=2):
            item = json.loads(line)
            if item.get("kind") != "record" or not isinstance(item.get("record"), dict):
                raise ValueError(f"invalid progress record at line {line_number}: {path}")
            record = item["record"]
            sample_id = record.get("sample_id")
            if not sample_id or sample_id in seen:
                raise ValueError(f"duplicate or missing sample_id at line {line_number}: {path}")
            seen.add(sample_id)
            records.append(record)
        return str(header["started_at"]), records

    path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header = {
        "kind": "header",
        "schema_version": 1,
        "started_at": started_at,
        "run": run_metadata,
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return started_at, []


def append_progress_record(path: Path, record: dict[str, Any]) -> None:
    item = {"kind": "record", "record": record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_official_turn_model(
    official_root: Path,
    config_path: Path,
    paraformer_model_dir: Path | None = None,
    sensevoice_model_dir: Path | None = None,
) -> Any:
    official_root = official_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    sys.path.insert(0, str(official_root))
    try:
        if paraformer_model_dir is not None:
            paraformer_model_dir = paraformer_model_dir.resolve(strict=True)
            asr_module = importlib.import_module("model.asr")

            class LocalParaformerASR:
                def __init__(self):
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks

                    self.asr_pipeline = pipeline(
                        task=Tasks.auto_speech_recognition,
                        model=str(paraformer_model_dir),
                        device="cuda",
                        disable_pbar=True,
                        disable_update=True,
                    )

                def recognize(self, audio_chunk, sample_rate=16_000):
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.mean(axis=1)
                    if sample_rate != 16_000:
                        audio_chunk = soxr.resample(audio_chunk, sample_rate, 16_000)
                    try:
                        return self.asr_pipeline(audio_chunk)[0]["text"].strip()
                    except Exception as exc:
                        print(f"ASR recognition failed: {exc}")
                        return ""

            asr_module.ParaformerASR = LocalParaformerASR
        if sensevoice_model_dir is not None:
            sensevoice_model_dir = sensevoice_model_dir.resolve(strict=True)
            asr_module = importlib.import_module("model.asr")

            class LocalSensevoiceASR:
                def __init__(self, language="auto"):
                    from funasr import AutoModel
                    from funasr.utils.postprocess_utils import (
                        rich_transcription_postprocess,
                    )

                    self.sensevoice_model = AutoModel(
                        model=str(sensevoice_model_dir),
                        trust_remote_code=False,
                        device="cuda",
                        disable_pbar=True,
                        disable_update=True,
                    )
                    self.language = language
                    remove_set = {
                        "😊",
                        "😔",
                        "😡",
                        "😰",
                        "🤢",
                        "😮",
                        "🎼",
                        "👏",
                        "😀",
                        "😭",
                        "🤧",
                        "😷",
                    }
                    self.pattern = "[" + "".join(remove_set) + "]"
                    self.rich_transcription_postprocess = (
                        rich_transcription_postprocess
                    )

                def clean_sensevoice_text(self, value: str) -> str:
                    if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", value):
                        return ""
                    return re.sub(self.pattern, "", value)

                def recognize(self, audio_chunk, sample_rate=16_000, language=None):
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.mean(axis=1)
                    if sample_rate != 16_000:
                        audio_chunk = soxr.resample(audio_chunk, sample_rate, 16_000)
                    if language is None:
                        language = self.language
                    try:
                        result = self.rich_transcription_postprocess(
                            self.sensevoice_model.generate(
                                input=audio_chunk,
                                cache={},
                                language=language,
                                use_itn=True,
                                batch_size=16,
                            )[0]["text"]
                        ).strip()
                        return self.clean_sensevoice_text(result)
                    except Exception as exc:
                        print(f"ASR recognition failed: {exc}")
                        return ""

            asr_module.SensevoiceASR = LocalSensevoiceASR
        service_model = importlib.import_module("service.model")
        turn_model = service_model.TurnModel(str(config_path))
    finally:
        if sys.path[0] == str(official_root):
            sys.path.pop(0)
    attach_raw_state_capture(turn_model)
    return turn_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official SoulX on Easy Turn with per-chunk audit output.")
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paraformer-model-dir", type=Path)
    parser.add_argument("--sensevoice-model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-silence-ms", type=int, default=1600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-label", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    samples = discover_samples(args.dataset_root, args.language)
    samples = select_sample_subset(samples, args.limit, args.limit_per_label)
    resolved_dataset_root = args.dataset_root.resolve(strict=True)
    resolved_official_root = args.official_root.resolve(strict=True)
    resolved_config = args.config.resolve(strict=True)
    resolved_paraformer = (
        args.paraformer_model_dir.resolve(strict=True)
        if args.paraformer_model_dir is not None
        else None
    )
    resolved_sensevoice = (
        args.sensevoice_model_dir.resolve(strict=True)
        if args.sensevoice_model_dir is not None
        else None
    )
    run_metadata = {
        "language": args.language,
        "dataset_root": str(resolved_dataset_root),
        "official_root": str(resolved_official_root),
        "config_path": str(resolved_config),
        "config_sha256": sha256_file(resolved_config),
        "benchmark_runner_sha256": sha256_file(Path(__file__)),
        "paraformer_model_dir": str(resolved_paraformer) if resolved_paraformer else None,
        "sensevoice_model_dir": str(resolved_sensevoice) if resolved_sensevoice else None,
        "tail_silence_ms": args.tail_silence_ms,
        "sample_count": len(samples),
        "sample_inventory_sha256": sha256_lines([sample.sample_id for sample in samples]),
    }
    progress_path = progress_path_for(output)
    started_at, records = load_or_create_progress(progress_path, run_metadata, args.resume)
    selected_ids = {sample.sample_id for sample in samples}
    completed_ids = {record["sample_id"] for record in records}
    if not completed_ids <= selected_ids:
        raise ValueError("progress journal contains samples outside the requested inventory")
    remaining = [sample for sample in samples if sample.sample_id not in completed_ids]
    resumed_record_count = len(records)
    turn_model = None
    if remaining:
        turn_model = load_official_turn_model(
            resolved_official_root,
            resolved_config,
            resolved_paraformer,
            resolved_sensevoice,
        )
    for sample in remaining:
        assert turn_model is not None
        record = evaluate_sample(turn_model, sample, args.tail_silence_ms)
        records.append(record)
        append_progress_record(progress_path, record)
        print(json.dumps({k: record[k] for k in ("sample_id", "label", "prediction", "correct")}, ensure_ascii=False))
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **run_metadata,
        "modelscope_cache": os.getenv("MODELSCOPE_CACHE"),
        "resumed_record_count": resumed_record_count,
        "summary": summarize(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    progress_path.unlink()
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
