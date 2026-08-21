"""Paired Table 3 comparisons for Stage 3 continuation checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from .continual_training import atomic_json_write, sha256_file, utc_now


CLASS_KEYS = (
    ("en", "complete"),
    ("en", "incomplete"),
    ("zh", "complete"),
    ("zh", "incomplete"),
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def exact_mcnemar_p_value(base_correct: list[bool], candidate_correct: list[bool]) -> float:
    improved = sum(not before and after for before, after in zip(base_correct, candidate_correct))
    regressed = sum(before and not after for before, after in zip(base_correct, candidate_correct))
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    smaller = min(improved, regressed)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_delta_ci(
    per_sample_deltas: list[float], seed: int, repetitions: int = 10_000
) -> list[float]:
    if not per_sample_deltas:
        raise ValueError("bootstrap input is empty")
    rng = random.Random(seed)
    size = len(per_sample_deltas)
    values = []
    for _ in range(repetitions):
        values.append(
            sum(per_sample_deltas[rng.randrange(size)] for _ in range(size)) / size
        )
    values.sort()
    return [
        100 * values[int(0.025 * repetitions)],
        100 * values[int(0.975 * repetitions) - 1],
    ]


def compare_class(
    baseline: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    before = baseline["records"]
    after = candidate["records"]
    if [item["sample_id"] for item in before] != [item["sample_id"] for item in after]:
        raise ValueError("baseline/candidate sample order mismatch")
    base_correct = [bool(item["correct"]) for item in before]
    candidate_correct = [bool(item["correct"]) for item in after]
    deltas = [float(after_value) - float(before_value) for before_value, after_value in zip(base_correct, candidate_correct)]
    total = len(deltas)
    base_count = sum(base_correct)
    candidate_count = sum(candidate_correct)
    improved = sum(not a and b for a, b in zip(base_correct, candidate_correct))
    regressed = sum(a and not b for a, b in zip(base_correct, candidate_correct))
    return {
        "total": total,
        "baseline_correct": base_count,
        "candidate_correct": candidate_count,
        "baseline_accuracy_percent": 100 * base_count / total,
        "candidate_accuracy_percent": 100 * candidate_count / total,
        "delta_percentage_points": 100 * (candidate_count - base_count) / total,
        "paired_improved_count": improved,
        "paired_regressed_count": regressed,
        "paired_unchanged_count": total - improved - regressed,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p_value(base_correct, candidate_correct),
        "paired_bootstrap_delta_95ci_percentage_points": bootstrap_delta_ci(deltas, seed),
        "per_sample_deltas": deltas,
    }


def zh_subgroups(payloads: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for label in ("complete", "incomplete"):
        payload = payloads[("zh", label)]
        for subgroup in ("real", "synthetic"):
            selected = []
            for record in payload["records"]:
                parts = Path(record["wav_path"]).parts
                is_real = "real" in parts
                if (subgroup == "real" and is_real) or (subgroup == "synthetic" and not is_real):
                    selected.append(record)
            result[f"{label}/{subgroup}"] = {
                "correct": sum(bool(item["correct"]) for item in selected),
                "total": len(selected),
                "accuracy_percent": 100 * sum(bool(item["correct"]) for item in selected) / len(selected),
            }
    return result


def summarize_checkpoint(
    baseline_payloads: dict[tuple[str, str], dict[str, Any]],
    candidate_payloads: dict[tuple[str, str], dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    continuation = gate["continuation_checkpoint"]
    step = int(continuation["local_step"])
    classes = {}
    raw_deltas = {}
    for class_index, key in enumerate(CLASS_KEYS):
        name = "/".join(key)
        comparison = compare_class(
            baseline_payloads[key], candidate_payloads[key], 42 + step * 10 + class_index
        )
        raw_deltas[key] = comparison.pop("per_sample_deltas")
        classes[name] = comparison
    languages = {}
    for language_index, language in enumerate(("en", "zh")):
        complete = classes[f"{language}/complete"]
        incomplete = classes[f"{language}/incomplete"]
        repetitions = 10_000
        rng = random.Random(4200 + step * 10 + language_index)
        complete_raw = raw_deltas[(language, "complete")]
        incomplete_raw = raw_deltas[(language, "incomplete")]
        bootstrapped = []
        for _ in range(repetitions):
            c_delta = sum(complete_raw[rng.randrange(len(complete_raw))] for _ in complete_raw) / len(complete_raw)
            i_delta = sum(incomplete_raw[rng.randrange(len(incomplete_raw))] for _ in incomplete_raw) / len(incomplete_raw)
            bootstrapped.append(0.5 * (c_delta + i_delta))
        bootstrapped.sort()
        baseline_macro = 0.5 * (
            complete["baseline_accuracy_percent"] + incomplete["baseline_accuracy_percent"]
        )
        candidate_macro = 0.5 * (
            complete["candidate_accuracy_percent"] + incomplete["candidate_accuracy_percent"]
        )
        languages[language] = {
            "baseline_macro_accuracy_percent": baseline_macro,
            "candidate_macro_accuracy_percent": candidate_macro,
            "delta_percentage_points": candidate_macro - baseline_macro,
            "paired_bootstrap_delta_95ci_percentage_points": [
                100 * bootstrapped[int(0.025 * repetitions)],
                100 * bootstrapped[int(0.975 * repetitions) - 1],
            ],
        }
    stable = all(
        languages[language]["delta_percentage_points"] >= -1.0
        for language in ("en", "zh")
    ) and all(item["delta_percentage_points"] >= -2.0 for item in classes.values())
    decline_trigger = any(
        languages[language]["delta_percentage_points"] < -3.0
        for language in ("en", "zh")
    ) or any(item["delta_percentage_points"] < -5.0 for item in classes.values())
    return {
        "local_step": step,
        "estimated_total_optimizer_step": continuation.get("estimated_total_optimizer_step"),
        "checkpoint": continuation,
        "evidence_gate_path": gate.get("path"),
        "classes": classes,
        "languages": languages,
        "zh_subgroups": zh_subgroups(candidate_payloads),
        "almost_unchanged": stable,
        "obvious_decline_trigger": decline_trigger,
        "obvious_decline_confirmed": False,
    }


def build_sweep_index(
    baseline_root: Path, sweep_root: Path, output: Path
) -> dict[str, Any]:
    baseline_root = baseline_root.resolve(strict=True)
    baseline_payloads = {
        key: load_json(baseline_root / f"{key[0]}-{key[1]}.json") for key in CLASS_KEYS
    }
    checkpoint_rows = []
    for gate_path in sorted(sweep_root.glob("step*/table3-gate.json")):
        gate = load_json(gate_path)
        if not gate.get("gate_passed") or gate.get("evaluation_mode") != "continuation":
            raise RuntimeError(f"invalid continuation gate: {gate_path}")
        candidate_payloads = {
            key: load_json(gate_path.parent / f"{key[0]}-{key[1]}.json")
            for key in CLASS_KEYS
        }
        gate["path"] = str(gate_path)
        row = summarize_checkpoint(baseline_payloads, candidate_payloads, gate)
        row["evidence_gate_sha256"] = sha256_file(gate_path)
        checkpoint_rows.append(row)
    checkpoint_rows.sort(key=lambda item: item["local_step"])
    for index in range(len(checkpoint_rows) - 1):
        if checkpoint_rows[index]["obvious_decline_trigger"] and checkpoint_rows[index + 1]["obvious_decline_trigger"]:
            checkpoint_rows[index]["obvious_decline_confirmed"] = True
    payload = {
        "schema_version": 1,
        "status": "complete" if checkpoint_rows else "pending",
        "updated_at_utc": utc_now(),
        "baseline_root": str(baseline_root),
        "baseline_gate_sha256": sha256_file(baseline_root / "table3-gate.json"),
        "primary_rule": "last-terminal-v1",
        "selection_used_paper_targets": False,
        "definitions": {
            "almost_unchanged": "EN and ZH macro delta >= -1pp and every class delta >= -2pp",
            "obvious_decline_trigger": "any language macro delta < -3pp or any class delta < -5pp",
            "obvious_decline_confirmed": "trigger also present at the next preregistered checkpoint",
        },
        "checkpoints": checkpoint_rows,
    }
    atomic_json_write(output, payload)
    return payload
