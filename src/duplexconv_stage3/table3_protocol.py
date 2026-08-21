"""Frozen candidate protocol helpers for SoulX-Duplug Table 3.

This module is intentionally independent from either SoulX runtime.  It owns
sample ordering, terminal-state readout, summaries, and immutable identities so
the inference runner and the gate use exactly the same label-independent logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 2
PRIMARY_RULE = "last-terminal-v1"
SENSITIVITY_RULES = (
    "first-terminal-v1",
    "closest-to-file-endpoint-v1",
    "first-at-or-after-file-endpoint-v1",
)
ALL_RULES = (PRIMARY_RULE, *SENSITIVITY_RULES)

EXPECTED_UPSTREAM_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
EXPECTED_UPSTREAM_SCRIPT_SHA256 = (
    "1558fffafdf1b0a5ea521a8cfc3feb3fce0c57ac9e34bf2795348c6f58c52d73"
)
EXPECTED_OFFICIAL_CHECKPOINT_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS = ("embed_tokens_func.weight",)

EXPECTED_DATASET_IDENTITY_SHA256 = {
    ("en", "complete"): "4a8a2823a93fec9e55c85706a95b878022868f74f58053d3d7892a9966ab4f5e",
    ("en", "incomplete"): "e61a135189990e774aff857f470deb9688966c57e4d769f93d9ad376a9010339",
    ("zh", "complete"): "f1462a85cbda4eb14c7a5ce2aa6e35d77215d48bd5d19cce7fc668d839d9bc2e",
    ("zh", "incomplete"): "86f34faded24e24be6d0afd8f0c42336984892c0d35d65ef1d018b59007dca7f",
}
EXPECTED_MODEL_MANIFEST_SHA256 = {
    "llm": "e83435f9aabb3455b3c5e43106f4d192b60e534c4230913c0e8f901ab74d0937",
    "speech_tokenizer": "b037935d145c0c940245d1e75439c40ef51938719fa91135ac3e78d8a34b5087",
    "asr_en": "fce9aa26033cebb152699a4c95ad6ac40598b441c2b130878a018e01f49b170a",
    "asr_zh": "e4314e9e73d4469a62c459cd76da2cb6777e46cd2389fc7958fff134042093fd",
}

EXPECTED_COUNTS = {
    ("en", "complete"): 318,
    ("en", "incomplete"): 299,
    ("zh", "complete"): 300,
    ("zh", "incomplete"): 300,
}
EXPECTED_SAMPLE_ORDER = {"en": "official-os-walk", "zh": "official-list"}

EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.6.0",
    "torchaudio": "2.6.0",
    "transformers": "4.55.0",
    "pytorch-lightning": "2.5.2",
    "funasr": "1.2.6",
    "modelscope": "1.28.2",
    "numpy": "1.24.4",
    "omegaconf": "2.3.0",
    "soundfile": "0.12.1",
    "soxr": "0.5.0.post1",
}

TERMINAL_TO_LABEL = {"speak": "complete", "wait": "incomplete"}


@dataclass(frozen=True)
class Table3Sample:
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


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_dataset_path(dataset_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe dataset path: {relative}")
    resolved_root = dataset_root.resolve(strict=True)
    resolved = (resolved_root / relative).resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"dataset path escapes root: {relative}")
    return resolved


def discover_samples(
    dataset_root: Path,
    language: str,
    label: str,
    sample_order: str,
) -> list[Table3Sample]:
    """Discover exactly one Table 3 class in its frozen traversal order."""
    if language not in {"en", "zh"}:
        raise ValueError(f"unsupported language: {language}")
    if label not in {"complete", "incomplete"}:
        raise ValueError(f"unsupported label: {label}")
    if sample_order != EXPECTED_SAMPLE_ORDER[language]:
        raise ValueError(
            f"unexpected sample order for {language}: {sample_order} != "
            f"{EXPECTED_SAMPLE_ORDER[language]}"
        )

    root = dataset_root.resolve(strict=True)
    paths: list[Path] = []
    if sample_order == "official-os-walk":
        for current_root, _directories, files in os.walk(root):
            for name in files:
                if not name.endswith(".wav"):
                    continue
                path = Path(current_root) / name
                relative = path.relative_to(root)
                if relative.parts and relative.parts[0] == label:
                    paths.append(path.resolve(strict=True))
    else:
        list_name = (
            "complete/complete_test.list"
            if label == "complete"
            else "incomplete/incomplete_real_test.list"
        )
        list_path = root / list_name
        if not list_path.is_file():
            raise FileNotFoundError(list_path)
        for line_number, line in enumerate(
            list_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                relative = Path(record["wav"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid list row {list_path}:{line_number}") from exc
            path = _safe_dataset_path(root, Path(str(relative).removeprefix("./")))
            if path.relative_to(root).parts[0] != label:
                raise ValueError(
                    f"label mismatch in {list_path}:{line_number}: {relative}"
                )
            paths.append(path)

    expected = EXPECTED_COUNTS[(language, label)]
    if len(paths) != expected:
        raise ValueError(
            f"unexpected Table 3 inventory for {language}/{label}: "
            f"{len(paths)} != {expected}"
        )
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate WAV paths for {language}/{label}")

    samples = []
    seen_ids = set()
    for path in paths:
        sample_id = f"{language}:{label}:{path.stem}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        samples.append(Table3Sample(sample_id, language, label, path))
    return samples


def terminal_states(trace: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in trace if item.get("state") in TERMINAL_TO_LABEL]


def classify_trace(
    trace: Sequence[dict[str, Any]],
    audio_duration_seconds: float,
    rule: str,
) -> dict[str, Any]:
    """Apply one frozen, label-independent readout rule to an upstream trace."""
    if rule not in ALL_RULES:
        raise ValueError(f"unsupported Table 3 rule: {rule}")
    terminals = terminal_states(trace)
    selected = None
    if terminals:
        if rule == "last-terminal-v1":
            selected = terminals[-1]
        elif rule == "first-terminal-v1":
            selected = terminals[0]
        elif rule == "closest-to-file-endpoint-v1":
            selected = min(
                terminals,
                key=lambda item: abs(
                    float(item["timestamp"][0]) - audio_duration_seconds
                ),
            )
        else:
            selected = next(
                (
                    item
                    for item in terminals
                    if float(item["timestamp"][0]) + 1e-9
                    >= audio_duration_seconds
                ),
                None,
            )
    return {
        "rule": rule,
        "prediction": (
            TERMINAL_TO_LABEL[selected["state"]] if selected is not None else "none"
        ),
        "selected_terminal": selected,
        "terminal_count": len(terminals),
    }


def make_inventory(samples: Sequence[Table3Sample]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "wav_path": str(sample.wav_path),
            "size": sample.wav_path.stat().st_size,
            "sha256": sha256_file(sample.wav_path),
        }
        for sample in samples
    ]


def portable_inventory_identity(
    inventory: Sequence[dict[str, Any]], dataset_root: Path
) -> str:
    """Hash ordered sample identity without tying it to an absolute mount path."""
    root = dataset_root.resolve(strict=True)
    rows = []
    for item in inventory:
        path = Path(item["wav_path"]).resolve(strict=True)
        if root not in path.parents:
            raise ValueError(f"inventory WAV escapes dataset root: {path}")
        rows.append(
            {
                "sample_id": item["sample_id"],
                "relative_path": path.relative_to(root).as_posix(),
                "size": item["size"],
                "sha256": item["sha256"],
            }
        )
    return canonical_sha256(rows)


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    for language in ("en", "zh"):
        for label in ("complete", "incomplete"):
            subset = [
                row
                for row in records
                if row["language"] == language and row["label"] == label
            ]
            if not subset:
                continue
            correct = sum(
                row["primary_readout"]["prediction"] == label for row in subset
            )
            key = f"{language}/{label}"
            by_class[key] = {
                "correct": correct,
                "total": len(subset),
                "accuracy": correct / len(subset),
                "predicted_none": sum(
                    row["primary_readout"]["prediction"] == "none"
                    for row in subset
                ),
            }
    return {"sample_count": len(records), "by_class": by_class}


def runtime_version_mismatches(packages: dict[str, str | None]) -> dict[str, Any]:
    mismatches = {}
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = packages.get(name)
        if actual != expected and not (
            name in {"torch", "torchaudio"}
            and isinstance(actual, str)
            and actual.startswith(expected + "+")
        ):
            mismatches[name] = {"expected": expected, "actual": actual}
    return mismatches
