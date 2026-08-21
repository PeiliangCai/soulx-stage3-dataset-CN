"""Strict evidence gate for four audited SoulX-Duplug Table 3 runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .table3_protocol import (
    EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS,
    EXPECTED_COUNTS,
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
    portable_inventory_identity,
    runtime_version_mismatches,
    sha256_file,
    summarize_records,
)
from .table3_reproduction import atomic_json_write, validate_trace


# Paper targets are intentionally isolated in the post-inference gate. The
# inference runner and trace classifier never import or receive these values.
PAPER_CORRECT = {
    ("en", "complete"): 247,
    ("en", "incomplete"): 266,
    ("zh", "complete"): 268,
    ("zh", "incomplete"): 238,
}


_VERIFIED_FILE_HASHES: dict[tuple[str, int, int], str] = {}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result root must be an object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: {actual!r} != {expected!r}")


def verify_file(path_value: str, expected_sha256: str, description: str) -> None:
    path = Path(path_value).resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    actual = _VERIFIED_FILE_HASHES.get(cache_key)
    if actual is None:
        actual = sha256_file(path)
        _VERIFIED_FILE_HASHES[cache_key] = actual
    if actual != expected_sha256:
        raise ValueError(
            f"{description} hash mismatch: {actual} != {expected_sha256}: {path}"
        )


def verify_directory_manifest(manifest: dict[str, Any], description: str) -> None:
    root = Path(manifest.get("root", ""))
    files = manifest.get("files")
    if not root.is_dir() or not isinstance(files, list) or not files:
        raise ValueError(f"invalid {description} manifest")
    require_equal(
        manifest.get("manifest_sha256"),
        canonical_sha256(files),
        f"{description} manifest hash",
    )
    require_equal(manifest.get("file_count"), len(files), f"{description} file count")
    require_equal(
        manifest.get("total_bytes"),
        sum(item["size"] for item in files),
        f"{description} total bytes",
    )
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in {description} manifest: {relative}")
        path = (root / relative).resolve(strict=True)
        if root.resolve(strict=True) not in path.parents:
            raise ValueError(f"escaped path in {description} manifest: {relative}")
        require_equal(path.stat().st_size, item["size"], f"{description} size")
        verify_file(str(path), item["sha256"], f"{description} artifact")


def verify_project_identity(project: dict[str, Any]) -> None:
    root = Path(project.get("root", "")).resolve(strict=True)
    commit = project.get("commit")
    require_equal(project.get("dirty"), False, "project dirty flag")
    require_equal(project.get("dirty_lines"), [], "project dirty lines")
    live_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    live_tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    live_dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    require_equal(live_commit, commit, "live project commit")
    require_equal(live_tree, project.get("tree"), "live project tree")
    require_equal(live_dirty, "", "live project worktree")
    runner_files = project.get("runner_files")
    if not isinstance(runner_files, list) or not runner_files:
        raise ValueError("project runner source manifest is missing")
    for item in runner_files:
        require_equal(item.get("tracked_in_head"), True, "runner source tracking")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe project runner path: {relative}")
        source = (root / relative).resolve(strict=True)
        if root not in source.parents:
            raise ValueError(f"project runner path escapes root: {relative}")
        verify_file(str(source), item["sha256"], "project runner source")


def audit_result(
    path: Path,
    language: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_json(path)
    require_equal(payload.get("schema_version"), SCHEMA_VERSION, "schema version")
    require_equal(payload.get("status"), "complete", "result status")
    require_equal(payload.get("run_mode"), "formal", "run mode")
    require_equal(payload.get("language"), language, "language")
    require_equal(payload.get("label"), label, "label")

    protocol = payload.get("protocol", {})
    require_equal(protocol.get("status"), "frozen-candidate-v1", "protocol status")
    require_equal(protocol.get("primary_rule"), PRIMARY_RULE, "primary rule")
    require_equal(
        protocol.get("sensitivity_rules"),
        list(SENSITIVITY_RULES),
        "sensitivity rules",
    )
    require_equal(protocol.get("post_silence_samples"), 32_000, "post silence")
    require_equal(protocol.get("post_silence_seconds"), 2.0, "post silence seconds")
    require_equal(protocol.get("far_field_filter"), False, "far-field filter")
    require_equal(
        protocol.get("sample_order"), EXPECTED_SAMPLE_ORDER[language], "sample order"
    )
    require_equal(protocol.get("resume_allowed"), False, "resume policy")
    require_equal(
        protocol.get("autocast_enabled_by_runner"), False, "runner autocast"
    )

    checkpoint = payload.get("checkpoint", {})
    require_equal(
        checkpoint.get("sha256"),
        EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
        "official checkpoint",
    )
    verify_file(
        checkpoint["path"], EXPECTED_OFFICIAL_CHECKPOINT_SHA256, "checkpoint"
    )
    load_audit = checkpoint.get("load_audit", {})
    require_equal(
        load_audit.get("policy"),
        "known-late-bound-embedding-alias-v1",
        "checkpoint load policy",
    )
    require_equal(load_audit.get("status"), "accepted", "checkpoint load status")
    require_equal(
        load_audit.get("missing_keys_at_load"), [], "checkpoint missing keys at load"
    )
    require_equal(
        load_audit.get("unexpected_keys_at_load"),
        list(EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS),
        "checkpoint unexpected keys at load",
    )
    require_equal(
        load_audit.get("allowed_unexpected_keys"),
        list(EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS),
        "checkpoint allowed unexpected keys",
    )
    require_equal(load_audit.get("final_missing_keys"), [], "final missing keys")
    require_equal(
        load_audit.get("final_unexpected_keys"), [], "final unexpected keys"
    )
    require_equal(load_audit.get("shape_mismatches"), {}, "checkpoint shape matches")
    require_equal(
        load_audit.get("checkpoint_alias_tied"), True, "checkpoint alias tying"
    )
    require_equal(
        load_audit.get("loaded_model_alias_tied"), True, "loaded model alias tying"
    )

    upstream = payload.get("upstream", {})
    require_equal(upstream.get("commit"), EXPECTED_UPSTREAM_COMMIT, "upstream commit")
    require_equal(upstream.get("dirty"), False, "upstream dirty flag")
    require_equal(
        upstream.get("inference_script_sha256"),
        EXPECTED_UPSTREAM_SCRIPT_SHA256,
        "upstream inference script",
    )
    verify_file(
        upstream["inference_script"],
        EXPECTED_UPSTREAM_SCRIPT_SHA256,
        "upstream inference script",
    )
    live_root = Path(upstream["root"])
    live_commit = subprocess.check_output(
        ["git", "-C", str(live_root), "rev-parse", "HEAD"], text=True
    ).strip()
    live_dirty = subprocess.check_output(
        ["git", "-C", str(live_root), "status", "--porcelain"], text=True
    ).strip()
    require_equal(live_commit, EXPECTED_UPSTREAM_COMMIT, "live upstream commit")
    require_equal(live_dirty, "", "live upstream worktree")

    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("project identity is missing")
    verify_project_identity(project)

    runtime = payload.get("runtime_environment", {})
    if not str(runtime.get("python", "")).startswith("3.10."):
        raise ValueError(f"unexpected Python version: {runtime.get('python')}")
    mismatches = runtime_version_mismatches(runtime.get("packages", {}))
    if mismatches:
        raise ValueError(f"runtime version drift: {mismatches}")
    require_equal(runtime.get("cuda", {}).get("available"), True, "CUDA availability")

    config = payload.get("config", {})
    verify_file(config["path"], config["sha256"], "evaluation config")
    model_artifacts = payload.get("model_artifacts")
    if not isinstance(model_artifacts, dict):
        raise ValueError("model artifact manifests are missing")
    verify_directory_manifest(model_artifacts["llm"], "LLM")
    verify_directory_manifest(model_artifacts["speech_tokenizer"], "speech tokenizer")
    verify_directory_manifest(payload.get("asr_model", {}), "teacher ASR")
    require_equal(
        model_artifacts["llm"].get("manifest_sha256"),
        EXPECTED_MODEL_MANIFEST_SHA256["llm"],
        "LLM frozen manifest",
    )
    require_equal(
        model_artifacts["speech_tokenizer"].get("manifest_sha256"),
        EXPECTED_MODEL_MANIFEST_SHA256["speech_tokenizer"],
        "speech tokenizer frozen manifest",
    )
    require_equal(
        payload["asr_model"].get("manifest_sha256"),
        EXPECTED_MODEL_MANIFEST_SHA256[f"asr_{language}"],
        "teacher ASR frozen manifest",
    )
    initialization_log = payload.get("initialization_log", {})
    verify_file(
        initialization_log["path"],
        initialization_log["sha256"],
        "initialization log",
    )
    initialization_text = Path(initialization_log["path"]).read_text(encoding="utf-8")
    expected_fallback_fragment = (
        'Unexpected key(s) in state_dict: "embed_tokens_func.weight"'
    )
    if (
        initialization_text.count("load_state_dict failed") != 1
        or expected_fallback_fragment not in initialization_text
    ):
        raise ValueError("checkpoint did not emit the audited embedding-alias fallback")
    asr_cache = payload.get("asr_cache", {})
    verify_file(asr_cache["path"], asr_cache["sha256"], "teacher ASR cache")
    asr_cache_rows = load_jsonl(Path(asr_cache["path"]))
    asr_cache_by_key = {}
    for row in asr_cache_rows:
        key = row.get("key")
        if not isinstance(key, str) or key in asr_cache_by_key:
            raise ValueError(f"invalid or duplicate teacher ASR cache key: {key!r}")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"invalid teacher ASR cache text for key: {key}")
        diagnostic = row.get("backend_diagnostic")
        if diagnostic is not None and (
            not isinstance(diagnostic, dict)
            or diagnostic.get("outcome") != "official-empty-string-fallback"
            or not isinstance(diagnostic.get("exception_type"), str)
            or not isinstance(diagnostic.get("message"), str)
        ):
            raise ValueError(f"invalid teacher ASR fallback evidence for key: {key}")
        require_equal(
            row.get("backend_identity"),
            asr_cache.get("backend_identity"),
            "teacher ASR backend identity",
        )
        asr_cache_by_key[key] = row

    inventory = payload.get("dataset", {}).get("inventory")
    if not isinstance(inventory, list):
        raise ValueError("dataset inventory is missing")
    require_equal(
        payload["dataset"].get("inventory_sha256"),
        canonical_sha256(inventory),
        "dataset inventory hash",
    )
    dataset_root = Path(payload["dataset"]["root"])
    portable_identity = portable_inventory_identity(inventory, dataset_root)
    require_equal(
        payload["dataset"].get("portable_identity_sha256"),
        portable_identity,
        "portable dataset identity",
    )
    require_equal(
        portable_identity,
        EXPECTED_DATASET_IDENTITY_SHA256[(language, label)],
        "frozen dataset identity",
    )
    require_equal(
        payload["dataset"].get("source_class_identity_sha256"),
        EXPECTED_DATASET_IDENTITY_SHA256[(language, label)],
        "source class dataset identity",
    )
    expected_count = EXPECTED_COUNTS[(language, label)]
    require_equal(len(inventory), expected_count, "dataset inventory count")
    inventory_ids = [item.get("sample_id") for item in inventory]
    if len(set(inventory_ids)) != len(inventory_ids):
        raise ValueError("dataset inventory contains duplicate sample IDs")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("records are missing")
    require_equal(len(records), expected_count, "record count")
    record_ids = [item.get("sample_id") for item in records]
    require_equal(record_ids, inventory_ids, "record sequence")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("records contain duplicate sample IDs")

    ordered_asr_calls = []
    for index, (record, item) in enumerate(zip(records, inventory)):
        prefix = f"record {index} ({record.get('sample_id')})"
        require_equal(record.get("sequence_index"), index, f"{prefix} sequence index")
        require_equal(record.get("language"), language, f"{prefix} language")
        require_equal(record.get("label"), label, f"{prefix} label")
        require_equal(record.get("wav_path"), item.get("wav_path"), f"{prefix} WAV path")
        require_equal(record.get("wav_size"), item.get("size"), f"{prefix} WAV size")
        require_equal(
            record.get("wav_sha256"), item.get("sha256"), f"{prefix} WAV hash"
        )
        verify_file(record["wav_path"], record["wav_sha256"], f"{prefix} WAV")
        trace = validate_trace(record.get("trace"))
        require_equal(
            canonical_sha256(trace),
            canonical_sha256(json.loads(Path(record["trace_path"]).read_text())),
            f"{prefix} trace content",
        )
        verify_file(
            record["trace_path"], record["trace_sha256"], f"{prefix} trace"
        )
        verify_file(record["log_path"], record["log_sha256"], f"{prefix} log")
        primary = classify_trace(
            trace, float(record["audio_duration_seconds"]), PRIMARY_RULE
        )
        require_equal(record.get("primary_readout"), primary, f"{prefix} primary")
        require_equal(
            record.get("correct"),
            primary["prediction"] == label,
            f"{prefix} correctness",
        )
        sensitivity = {
            rule: classify_trace(trace, float(record["audio_duration_seconds"]), rule)
            for rule in SENSITIVITY_RULES
        }
        require_equal(
            record.get("sensitivity_readouts"), sensitivity, f"{prefix} sensitivity"
        )
        calls = record.get("teacher_asr_calls")
        if not isinstance(calls, list):
            raise ValueError(f"{prefix} teacher ASR calls are missing")
        for call in calls:
            if not isinstance(call.get("key"), str) or not isinstance(
                call.get("text"), str
            ):
                raise ValueError(f"{prefix} has invalid teacher ASR provenance")
            ordered_asr_calls.append(call)
        forward_audit = record.get("llm_forward_audit")
        if not isinstance(forward_audit, list) or not forward_audit:
            raise ValueError(f"{prefix} LLM forward audit is missing")
        for forward_index, forward in enumerate(forward_audit):
            require_equal(
                forward.get("forward_index"),
                forward_index,
                f"{prefix} LLM forward index",
            )
            logits = forward.get("state_logits")
            if set(logits or {}) != {
                "complete",
                "incomplete",
                "backchannel",
                "idle",
                "nonidle",
            } or not all(math.isfinite(float(value)) for value in logits.values()):
                raise ValueError(f"{prefix} contains invalid state logits")

    seen_asr_keys = set()
    observed_hits = 0
    observed_misses = 0
    for call in ordered_asr_calls:
        key = call["key"]
        expected_hit = key in seen_asr_keys
        require_equal(call.get("cache_hit"), expected_hit, "teacher ASR cache-hit order")
        cached = asr_cache_by_key.get(key)
        if cached is None:
            raise ValueError(f"teacher ASR call is absent from cache JSONL: {key}")
        require_equal(call["text"], cached["text"], "teacher ASR cached text")
        require_equal(
            call.get("backend_diagnostic"),
            cached.get("backend_diagnostic"),
            "teacher ASR fallback evidence",
        )
        if expected_hit:
            observed_hits += 1
        else:
            observed_misses += 1
            seen_asr_keys.add(key)
    require_equal(set(asr_cache_by_key), seen_asr_keys, "teacher ASR cache key set")
    require_equal(asr_cache.get("entries"), len(asr_cache_rows), "ASR cache entries")
    require_equal(asr_cache.get("misses"), observed_misses, "ASR cache misses")
    require_equal(asr_cache.get("hits"), observed_hits, "ASR cache hits")
    require_equal(
        asr_cache.get("fallback_entries"),
        sum(row.get("backend_diagnostic") is not None for row in asr_cache_rows),
        "ASR cache fallback entries",
    )

    recomputed_summary = summarize_records(records)
    require_equal(payload.get("summary"), recomputed_summary, "summary")
    class_summary = recomputed_summary["by_class"][f"{language}/{label}"]
    paper_correct = PAPER_CORRECT[(language, label)]
    delta_percentage_points = 100 * (
        class_summary["correct"] - paper_correct
    ) / expected_count
    report = {
        "result_path": str(path.resolve(strict=True)),
        "language": language,
        "label": label,
        "actual_correct": class_summary["correct"],
        "total": expected_count,
        "accuracy_percent": 100 * class_summary["accuracy"],
        "paper_correct": paper_correct,
        "paper_accuracy_percent": 100 * paper_correct / expected_count,
        "delta_percentage_points": delta_percentage_points,
        "within_one_percentage_point": abs(delta_percentage_points) <= 1.0 + 1e-12,
    }
    return payload, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and gate four Table 3 runs.")
    parser.add_argument("--en-complete", type=Path, required=True)
    parser.add_argument("--en-incomplete", type=Path, required=True)
    parser.add_argument("--zh-complete", type=Path, required=True)
    parser.add_argument("--zh-incomplete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite gate report: {args.output}")
    paths = {
        ("en", "complete"): args.en_complete,
        ("en", "incomplete"): args.en_incomplete,
        ("zh", "complete"): args.zh_complete,
        ("zh", "incomplete"): args.zh_incomplete,
    }
    payloads = {}
    reports = []
    for key, path in paths.items():
        payload, report = audit_result(path, *key)
        payloads[key] = payload
        reports.append(report)

    require_equal(
        len({item["checkpoint"]["sha256"] for item in payloads.values()}),
        1,
        "checkpoint identity count",
    )
    require_equal(
        len({item["upstream"]["tree"] for item in payloads.values()}),
        1,
        "upstream tree identity count",
    )
    require_equal(
        len({item["project"]["commit"] for item in payloads.values()}),
        1,
        "project commit identity count",
    )
    require_equal(
        len({item["project"]["tree"] for item in payloads.values()}),
        1,
        "project tree identity count",
    )
    require_equal(
        len({item["run_id"] for item in payloads.values()}),
        1,
        "run ID count",
    )
    cache_paths = [item["asr_cache"]["path"] for item in payloads.values()]
    require_equal(len(set(cache_paths)), 4, "fresh per-class ASR cache count")
    for language in ("en", "zh"):
        first = payloads[(language, "complete")]
        second = payloads[(language, "incomplete")]
        require_equal(
            first["config"]["sha256"],
            second["config"]["sha256"],
            f"{language} config identity",
        )
        require_equal(
            first["asr_model"]["manifest_sha256"],
            second["asr_model"]["manifest_sha256"],
            f"{language} ASR artifact identity",
        )
        require_equal(
            first.get("model_artifacts"),
            second.get("model_artifacts"),
            f"{language} model artifact identity",
        )

    gate_passed = all(item["within_one_percentage_point"] for item in reports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "frozen-candidate-v1",
        "criterion": "all four class accuracies within 1.0 percentage point",
        "evidence_audit_passed": True,
        "accuracy_gate_passed": gate_passed,
        "gate_passed": gate_passed,
        "continued_training_authorized": gate_passed,
        "checks": reports,
    }
    atomic_json_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
