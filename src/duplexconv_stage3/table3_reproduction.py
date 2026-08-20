"""Audited runner for the frozen SoulX-Duplug Table 3 candidate protocol."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from .table3_protocol import (
    EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS,
    EXPECTED_DATASET_IDENTITY_SHA256,
    EXPECTED_MODEL_MANIFEST_SHA256,
    EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
    EXPECTED_SAMPLE_ORDER,
    EXPECTED_UPSTREAM_COMMIT,
    EXPECTED_UPSTREAM_SCRIPT_SHA256,
    PRIMARY_RULE,
    SCHEMA_VERSION,
    SENSITIVITY_RULES,
    canonical_sha256,
    classify_trace,
    discover_samples,
    make_inventory,
    portable_inventory_identity,
    runtime_version_mismatches,
    sha256_file,
    summarize_records,
)


POST_SILENCE_SAMPLES = 32_000
POST_SILENCE_SECONDS = 2.0
RUNTIME_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "pytorch-lightning",
    "funasr",
    "modelscope",
    "numpy",
    "omegaconf",
    "soundfile",
    "soxr",
)
PROJECT_RUNNER_FILES = (
    "scripts/run_table3_reproduction.py",
    "src/duplexconv_stage3/table3_protocol.py",
    "src/duplexconv_stage3/table3_reproduction.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def collect_runtime_environment() -> dict[str, Any]:
    import torch

    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name_0": torch.cuda.get_device_name(0) if cuda_available else None,
            "capability_0": (
                list(torch.cuda.get_device_capability(0)) if cuda_available else None
            ),
        },
    }


def verify_runtime_environment(environment: dict[str, Any]) -> None:
    if not environment["cuda"]["available"]:
        raise RuntimeError("CUDA is required for the formal Table 3 reproduction")
    if not environment["python"].startswith("3.10."):
        raise RuntimeError(f"Python 3.10 is required: {environment['python']}")
    mismatches = runtime_version_mismatches(environment["packages"])
    if mismatches:
        raise RuntimeError(
            "runtime does not match the frozen Table 3 environment: "
            + json.dumps(mismatches, sort_keys=True)
        )


def verify_upstream_identity(official_root: Path) -> dict[str, Any]:
    root = official_root.resolve(strict=True)
    script = root / "scripts/duplex_inference.py"
    commit = git_output(root, "rev-parse", "HEAD")
    tree = git_output(root, "rev-parse", "HEAD^{tree}")
    dirty_lines = git_output(root, "status", "--porcelain").splitlines()
    script_sha256 = sha256_file(script)
    if commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError(f"unexpected upstream commit: {commit}")
    if dirty_lines:
        raise RuntimeError(f"upstream worktree is dirty: {dirty_lines}")
    if script_sha256 != EXPECTED_UPSTREAM_SCRIPT_SHA256:
        raise RuntimeError(f"unexpected upstream inference hash: {script_sha256}")
    return {
        "root": str(root),
        "commit": commit,
        "tree": tree,
        "dirty": False,
        "inference_script": str(script),
        "inference_script_sha256": script_sha256,
    }


def collect_project_identity(require_clean: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = git_output(root, "rev-parse", "HEAD")
    tree = git_output(root, "rev-parse", "HEAD^{tree}")
    dirty_lines = git_output(
        root, "status", "--porcelain", "--untracked-files=all"
    ).splitlines()
    files = []
    for relative_value in PROJECT_RUNNER_FILES:
        path = root / relative_value
        if not path.is_file():
            raise FileNotFoundError(f"missing project runner source: {path}")
        tracked = (
            subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", relative_value],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        files.append(
            {
                "path": relative_value,
                "sha256": sha256_file(path),
                "tracked_in_head": tracked,
            }
        )
    if require_clean and dirty_lines:
        raise RuntimeError(f"formal run requires a clean project worktree: {dirty_lines}")
    if require_clean and not all(item["tracked_in_head"] for item in files):
        raise RuntimeError("formal runner sources must be tracked in the project commit")
    return {
        "root": str(root),
        "commit": commit,
        "tree": tree,
        "dirty": bool(dirty_lines),
        "dirty_lines": dirty_lines,
        "runner_files": files,
    }


def directory_manifest(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    files = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"empty model artifact directory: {resolved}")
    return {
        "root": str(resolved),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "manifest_sha256": canonical_sha256(files),
        "files": files,
    }


class StrictParaformerASR:
    def __init__(self, model_dir: Path):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=str(model_dir),
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )

    def recognize(self, audio_chunk, sample_rate=16_000, **_kwargs):
        import numpy as np
        import soxr

        audio = np.asarray(audio_chunk)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            audio = soxr.resample(audio, sample_rate, 16_000)
        result = self.pipeline(audio)
        return result[0]["text"].strip()


class StrictSensevoiceASR:
    def __init__(self, model_dir: Path, language: str = "en"):
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.model = AutoModel(
            model=str(model_dir),
            trust_remote_code=False,
            device="cuda",
            disable_pbar=True,
            disable_update=True,
        )
        self.language = language
        self.postprocess = rich_transcription_postprocess
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

    def recognize(self, audio_chunk, sample_rate=16_000, language=None, **_kwargs):
        import numpy as np
        import soxr

        audio = np.asarray(audio_chunk)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            audio = soxr.resample(audio, sample_rate, 16_000)
        selected_language = language or self.language
        value = self.postprocess(
            self.model.generate(
                input=audio,
                cache={},
                language=selected_language,
                use_itn=True,
                batch_size=16,
            )[0]["text"]
        ).strip()
        if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", value):
            return ""
        return re.sub(self.pattern, "", value)


class FreshAuditedASRCache:
    """Fresh per-class ASR cache that records every text consumed by SoulX."""

    def __init__(
        self,
        path: Path,
        backend_factory: Callable[[], Any],
        backend_identity: str,
    ) -> None:
        if path.exists():
            raise FileExistsError(f"formal ASR cache must be fresh: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.backend_factory = backend_factory
        self.backend_identity = backend_identity
        self.backend = None
        self.values: dict[str, str] = {}
        self.sample_calls: list[dict[str, Any]] = []
        self.hits = 0
        self.misses = 0

    def initialize(self) -> None:
        if self.backend is None:
            self.backend = self.backend_factory()

    def begin_sample(self) -> None:
        self.sample_calls = []

    def recognize(self, audio, sample_rate=16_000, **kwargs):
        import numpy as np

        array = np.ascontiguousarray(audio)
        digest = hashlib.sha256()
        digest.update(self.backend_identity.encode("utf-8"))
        digest.update(str(sample_rate).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        key = digest.hexdigest()
        cache_hit = key in self.values
        if cache_hit:
            self.hits += 1
            text = self.values[key]
        else:
            self.misses += 1
            self.initialize()
            text = self.backend.recognize(
                audio, sample_rate=sample_rate, **kwargs
            )
            if not isinstance(text, str):
                raise TypeError("teacher ASR returned non-text output")
            self.values[key] = text
            row = {
                "key": key,
                "text": text,
                "backend_identity": self.backend_identity,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        call = {
            "key": key,
            "text": text,
            "cache_hit": cache_hit,
            "sample_rate": sample_rate,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        self.sample_calls.append(call)
        return text

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "backend_identity": self.backend_identity,
            "entries": len(self.values),
            "hits": self.hits,
            "misses": self.misses,
            "sha256": sha256_file(self.path) if self.path.is_file() else None,
        }


def load_upstream_module(official_root: Path):
    path = official_root / "scripts/duplex_inference.py"
    sys.path.insert(0, str(official_root))
    specification = importlib.util.spec_from_file_location(
        "audited_upstream_duplex_inference", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import upstream inference: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def model_dtype_manifest(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        counts[name] = counts.get(name, 0) + parameter.numel()
    return dict(sorted(counts.items()))


def audit_checkpoint_compatibility(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Accept only the official checkpoint's known late-bound embedding alias.

    The pinned upstream class assigns ``embed_tokens_func`` after it loads the
    checkpoint.  Exported checkpoints therefore contain this alias, while the
    object does not yet expose it at load time.  The canonical embedding and LM
    head are tied to the same tensor, so this is safe only when all three aliases
    remain tied and every other key and shape matches exactly.
    """
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        mmap=True,
        weights_only=False,
    )
    checkpoint_state = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    model_state = model.state_dict()
    alias_key = EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS[0]
    canonical_keys = (
        "llm.base_model.model.model.embed_tokens.weight",
        "llm.base_model.model.lm_head.weight",
    )

    final_missing = sorted(set(model_state) - set(checkpoint_state))
    final_unexpected = sorted(set(checkpoint_state) - set(model_state))
    shape_mismatches = {
        key: {
            "checkpoint": list(checkpoint_state[key].shape),
            "model": list(model_state[key].shape),
        }
        for key in sorted(set(model_state) & set(checkpoint_state))
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    }

    # At the exact load point, the only absent module is the late-bound alias.
    load_point_model_keys = set(model_state) - {alias_key}
    missing_at_load = sorted(load_point_model_keys - set(checkpoint_state))
    unexpected_at_load = sorted(set(checkpoint_state) - load_point_model_keys)
    expected_unexpected = list(EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS)

    required = (alias_key, *canonical_keys)
    if any(key not in checkpoint_state or key not in model_state for key in required):
        raise RuntimeError("checkpoint embedding aliases are incomplete")

    checkpoint_alias_tied = all(
        checkpoint_state[alias_key].data_ptr() == checkpoint_state[key].data_ptr()
        and checkpoint_state[alias_key].storage_offset()
        == checkpoint_state[key].storage_offset()
        and tuple(checkpoint_state[alias_key].stride())
        == tuple(checkpoint_state[key].stride())
        for key in canonical_keys
    )
    loaded_model_alias_tied = all(
        model_state[alias_key].data_ptr() == model_state[key].data_ptr()
        and model_state[alias_key].storage_offset() == model_state[key].storage_offset()
        and tuple(model_state[alias_key].stride()) == tuple(model_state[key].stride())
        for key in canonical_keys
    )
    accepted = (
        missing_at_load == []
        and unexpected_at_load == expected_unexpected
        and final_missing == []
        and final_unexpected == []
        and shape_mismatches == {}
        and checkpoint_alias_tied
        and loaded_model_alias_tied
    )
    result = {
        "policy": "known-late-bound-embedding-alias-v1",
        "status": "accepted" if accepted else "rejected",
        "missing_keys_at_load": missing_at_load,
        "unexpected_keys_at_load": unexpected_at_load,
        "allowed_unexpected_keys": expected_unexpected,
        "final_missing_keys": final_missing,
        "final_unexpected_keys": final_unexpected,
        "shape_mismatches": shape_mismatches,
        "checkpoint_state_key_count": len(checkpoint_state),
        "final_model_state_key_count": len(model_state),
        "checkpoint_alias_tied": checkpoint_alias_tied,
        "loaded_model_alias_tied": loaded_model_alias_tied,
        "canonical_alias_keys": list(canonical_keys),
    }
    del checkpoint
    if not accepted:
        raise RuntimeError(
            "official checkpoint compatibility audit failed: "
            + json.dumps(result, sort_keys=True)
        )
    return result


def attach_llm_logit_audit(model: Any) -> None:
    """Capture state-token logits for every upstream LLM forward call."""
    import torch

    original_forward = model.llm.forward
    token_ids = {
        "complete": int(model.config.model_config.user_complete_token_id),
        "incomplete": int(model.config.model_config.user_incomplete_token_id),
        "backchannel": int(model.config.model_config.user_backchannel_token_id),
        "idle": int(model.config.model_config.user_idle_token_id),
        "nonidle": int(model.config.model_config.user_nonidle_token_id),
    }

    def audited_forward(*args: Any, **kwargs: Any):
        output = original_forward(*args, **kwargs)
        final_logits = output.logits[0, -1]
        values = {
            name: float(final_logits[token_id].detach().float().cpu())
            for name, token_id in token_ids.items()
        }
        prediction_id = int(torch.argmax(final_logits).detach().cpu())
        model._table3_forward_audit.append(
            {
                "forward_index": len(model._table3_forward_audit),
                "state_logits": values,
                "argmax_token_id": prediction_id,
            }
        )
        return output

    model.llm.forward = audited_forward
    model._table3_forward_audit = []


def validate_trace(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, list) or not trace:
        raise ValueError("upstream state trace is empty or invalid")
    valid_states = {"speak", "wait", "backchannel", "idle", "nonidle", "unknown"}
    for index, item in enumerate(trace):
        if not isinstance(item, dict) or item.get("state") not in valid_states:
            raise ValueError(f"invalid state at trace index {index}: {item}")
        timestamp = item.get("timestamp")
        expected = [index * 0.16, (index + 1) * 0.16]
        if (
            not isinstance(timestamp, list)
            or len(timestamp) != 2
            or abs(float(timestamp[0]) - expected[0]) > 1e-8
            or abs(float(timestamp[1]) - expected[1]) > 1e-8
        ):
            raise ValueError(f"invalid timestamp at trace index {index}: {timestamp}")
    return trace


def evaluate_one(
    upstream: Any,
    cfg: Any,
    model: Any,
    asr_cache: FreshAuditedASRCache,
    sample: Any,
    inventory_row: dict[str, Any],
    trace_dir: Path,
    sequence_index: int,
) -> dict[str, Any]:
    import soundfile as sf

    safe_name = f"{sequence_index:04d}-{sample.wav_path.stem}"
    staged_wav = trace_dir / f"{safe_name}.wav"
    state_path = trace_dir / f"{safe_name}_states.json"
    log_path = trace_dir / f"{safe_name}.log"
    for path in (staged_wav, state_path, log_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"stale Table 3 artifact: {path}")
    staged_wav.symlink_to(sample.wav_path.resolve(strict=True))
    if staged_wav.resolve(strict=True) != sample.wav_path.resolve(strict=True):
        raise RuntimeError(f"staged WAV target mismatch: {staged_wav}")

    asr_cache.begin_sample()
    model._table3_forward_audit = []
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
            log_handle
        ):
            upstream.duplex_predict_160_cascade_asr(
                cfg, model, str(staged_wav), asr_cache
            )
    elapsed = time.perf_counter() - started
    if not state_path.is_file():
        raise RuntimeError(f"upstream did not write state trace: {state_path}")
    trace = validate_trace(json.loads(state_path.read_text(encoding="utf-8")))
    info = sf.info(sample.wav_path)
    duration = info.frames / info.samplerate
    primary = classify_trace(trace, duration, PRIMARY_RULE)
    sensitivity = {
        rule: classify_trace(trace, duration, rule) for rule in SENSITIVITY_RULES
    }
    return {
        "sequence_index": sequence_index,
        "sample_id": sample.sample_id,
        "language": sample.language,
        "label": sample.label,
        "wav_path": str(sample.wav_path),
        "wav_size": inventory_row["size"],
        "wav_sha256": inventory_row["sha256"],
        "audio_duration_seconds": duration,
        "source_sample_rate": info.samplerate,
        "source_channels": info.channels,
        "trace_path": str(state_path),
        "trace_sha256": sha256_file(state_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "trace": trace,
        "teacher_asr_calls": list(asr_cache.sample_calls),
        "llm_forward_audit": list(model._table3_forward_audit),
        "primary_readout": primary,
        "sensitivity_readouts": sensitivity,
        "correct": primary["prediction"] == sample.label,
        "runtime_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one class of the frozen, audited SoulX Table 3 candidate protocol."
    )
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--label", choices=("complete", "incomplete"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        help="Run a non-gating prefix only; omission is required for a formal run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.diagnostic_limit is not None and args.diagnostic_limit <= 0:
        raise ValueError("--diagnostic-limit must be positive")
    output = args.output.absolute()
    trace_dir = args.trace_dir.absolute()
    cache_path = args.asr_cache.absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if trace_dir.exists():
        raise FileExistsError(f"trace directory must be new: {trace_dir}")
    if cache_path.exists():
        raise FileExistsError(f"ASR cache must be new: {cache_path}")

    environment = collect_runtime_environment()
    verify_runtime_environment(environment)
    official_identity = verify_upstream_identity(args.official_root)
    config_path = args.config.resolve(strict=True)
    asr_model_dir = args.asr_model_dir.resolve(strict=True)
    all_samples = discover_samples(
        args.dataset_root,
        args.language,
        args.label,
        EXPECTED_SAMPLE_ORDER[args.language],
    )
    run_mode = "formal" if args.diagnostic_limit is None else "diagnostic"
    project_identity = collect_project_identity(require_clean=run_mode == "formal")
    full_inventory = make_inventory(all_samples)
    source_class_identity_sha256 = portable_inventory_identity(
        full_inventory, args.dataset_root
    )
    expected_dataset_identity = EXPECTED_DATASET_IDENTITY_SHA256[
        (args.language, args.label)
    ]
    if source_class_identity_sha256 != expected_dataset_identity:
        raise RuntimeError(
            "Table 3 dataset identity drift: "
            f"{source_class_identity_sha256} != {expected_dataset_identity}"
        )
    samples = (
        all_samples[: args.diagnostic_limit]
        if args.diagnostic_limit is not None
        else all_samples
    )
    inventory = full_inventory[: len(samples)]
    evaluated_inventory_identity_sha256 = portable_inventory_identity(
        inventory, args.dataset_root
    )
    trace_dir.mkdir(parents=True)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import WhisperFeatureExtractor

    upstream_root = Path(official_identity["root"])
    upstream = load_upstream_module(upstream_root)
    from config.config import RunConfig
    from models.state_prediction_model import State_Prediction_Model

    cfg = OmegaConf.merge(RunConfig(), OmegaConf.load(config_path))
    if int(cfg.infer_config.seed) != 42:
        raise RuntimeError("Table 3 seed must be 42")
    expected_config = {
        "chunk_size": 2560,
        "audio_back_size": 15360,
        "audio_ahead_size": 640,
        "sample_rate": 16000,
        "chunk_token_len_small": 2,
        "max_wait_num": 5,
        "max_mistake_num": 5,
        "single_round": False,
        "precision_field": "bf16",
        "enable_cascade_asr": True,
        "asr_model_name": "sensevoice" if args.language == "en" else "paraformer",
        "asr_language": args.language,
    }
    actual_config = {
        "chunk_size": int(cfg.infer_config.input.chunk_size),
        "audio_back_size": int(cfg.infer_config.input.audio_back_size),
        "audio_ahead_size": int(cfg.infer_config.input.audio_ahead_size),
        "sample_rate": int(cfg.infer_config.input.sample_rate),
        "chunk_token_len_small": int(cfg.infer_config.input.chunk_token_len_small),
        "max_wait_num": int(cfg.infer_config.max_wait_num),
        "max_mistake_num": int(cfg.infer_config.max_mistake_num),
        "single_round": bool(cfg.infer_config.single_round),
        "precision_field": str(cfg.infer_config.precision),
        "enable_cascade_asr": bool(cfg.model_config.enable_cascade_asr),
        "asr_model_name": str(cfg.infer_config.asr.model_name),
        "asr_language": str(cfg.infer_config.asr.language),
    }
    if actual_config != expected_config:
        raise RuntimeError(
            f"Table 3 config drift: {actual_config} != {expected_config}"
        )

    checkpoint = Path(cfg.model_config.init_ckpt_path_lora).resolve(strict=True)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(f"unexpected official checkpoint: {checkpoint_sha256}")
    model_artifacts = {
        "llm": directory_manifest(Path(cfg.model_config.model_name)),
        "speech_tokenizer": directory_manifest(
            Path(cfg.model_config.glm_tokenizer_path)
        ),
    }
    for name, manifest in model_artifacts.items():
        expected_manifest = EXPECTED_MODEL_MANIFEST_SHA256[name]
        if manifest["manifest_sha256"] != expected_manifest:
            raise RuntimeError(
                f"{name} model artifact drift: "
                f"{manifest['manifest_sha256']} != {expected_manifest}"
            )

    upstream.pl.seed_everything(int(cfg.infer_config.seed))
    torch.cuda.manual_seed(int(cfg.infer_config.seed))
    torch.manual_seed(int(cfg.infer_config.seed))
    np.random.seed(int(cfg.infer_config.seed))
    random.seed(int(cfg.infer_config.seed))

    initialization_log = trace_dir / "initialization.log"
    with initialization_log.open("x", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
            log_handle
        ):
            model = State_Prediction_Model(cfg)
            model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                cfg.model_config.glm_tokenizer_path
            )
            model.eval().to("cuda")
    initialization_text = initialization_log.read_text(encoding="utf-8")
    checkpoint_load_audit = audit_checkpoint_compatibility(model, checkpoint)
    expected_fallback_fragment = (
        'Unexpected key(s) in state_dict: "embed_tokens_func.weight"'
    )
    if (
        initialization_text.count("load_state_dict failed") != 1
        or expected_fallback_fragment not in initialization_text
    ):
        raise RuntimeError(
            "official checkpoint did not emit the pinned late-bound alias fallback; "
            "see initialization log"
        )
    effective_dtypes = model_dtype_manifest(model)
    attach_llm_logit_audit(model)

    asr_manifest = directory_manifest(asr_model_dir)
    expected_asr_manifest = EXPECTED_MODEL_MANIFEST_SHA256[f"asr_{args.language}"]
    if asr_manifest["manifest_sha256"] != expected_asr_manifest:
        raise RuntimeError(
            "teacher ASR artifact drift: "
            f"{asr_manifest['manifest_sha256']} != {expected_asr_manifest}"
        )
    backend_identity = (
        f"{args.language}:{asr_manifest['manifest_sha256']}:strict-local-v1"
    )
    backend_factory: Callable[[], Any]
    if args.language == "zh":
        backend_factory = lambda: StrictParaformerASR(asr_model_dir)
    else:
        backend_factory = lambda: StrictSensevoiceASR(asr_model_dir, "en")
    asr_cache = FreshAuditedASRCache(
        cache_path, backend_factory, backend_identity
    )
    with initialization_log.open("a", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
            log_handle
        ):
            asr_cache.initialize()

    protocol = {
        "status": "frozen-candidate-v1",
        "inference_core": (
            "untouched training-code "
            "scripts.duplex_inference.duplex_predict_160_cascade_asr"
        ),
        "primary_rule": PRIMARY_RULE,
        "sensitivity_rules": list(SENSITIVITY_RULES),
        "post_silence_samples": POST_SILENCE_SAMPLES,
        "post_silence_seconds": POST_SILENCE_SECONDS,
        "far_field_filter": False,
        "sample_order": EXPECTED_SAMPLE_ORDER[args.language],
        "rng_scope": "fresh process and seed for exactly one language/class",
        "resume_allowed": False,
        "config_precision_field": str(cfg.infer_config.precision),
        "autocast_enabled_by_runner": False,
        "effective_model_parameter_dtypes": effective_dtypes,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "run_mode": run_mode,
        "run_id": args.run_id,
        "started_at_utc": utc_now(),
        "language": args.language,
        "label": args.label,
        "protocol": protocol,
        "checkpoint": {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "load_audit": checkpoint_load_audit,
        },
        "upstream": official_identity,
        "project": project_identity,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "effective": actual_config,
        },
        "model_artifacts": model_artifacts,
        "dataset": {
            "root": str(args.dataset_root.resolve(strict=True)),
            "inventory": inventory,
            "inventory_sha256": canonical_sha256(inventory),
            "portable_identity_sha256": evaluated_inventory_identity_sha256,
            "source_class_identity_sha256": source_class_identity_sha256,
        },
        "asr_model": asr_manifest,
        "runtime_environment": environment,
        "initialization_log": {
            "path": str(initialization_log),
            "sha256": sha256_file(initialization_log),
        },
        "records": [],
    }
    atomic_json_write(output, payload)

    inventory_by_id = {item["sample_id"]: item for item in inventory}
    for index, sample in enumerate(samples):
        record = evaluate_one(
            upstream,
            cfg,
            model,
            asr_cache,
            sample,
            inventory_by_id[sample.sample_id],
            trace_dir,
            index,
        )
        payload["records"].append(record)
        payload["updated_at_utc"] = utc_now()
        payload["partial_summary"] = summarize_records(payload["records"])
        atomic_json_write(output, payload)
        print(
            json.dumps(
                {
                    "sample_id": record["sample_id"],
                    "prediction": record["primary_readout"]["prediction"],
                    "correct": record["correct"],
                    "progress": f"{index + 1}/{len(samples)}",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    payload["status"] = "complete"
    payload["completed_at_utc"] = utc_now()
    payload["summary"] = summarize_records(payload["records"])
    payload["asr_cache"] = asr_cache.stats()
    payload.pop("partial_summary", None)
    atomic_json_write(output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
