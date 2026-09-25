#!/usr/bin/env python3
"""Run one approved DuplexConv shard through the frozen Stage 3 data pipeline.

This is an orchestration-only entrypoint.  It invokes the already-frozen
download, leakage, state-label, ASR, tokenizer, export, validation, and Gate D
implementations without changing their processing semantics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
from typing import Any
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/autodl-tmp/conda_envs/soulx-duplug-official/bin/python")
DATA_ROOT = Path("/root/autodl-tmp/dataset/duplexconv")
METADATA_ARCHIVE = DATA_ROOT / "raw/metadata/jsons.tar.gz"
BENCHMARK_RECORDS = Path(
    "/root/autodl-tmp/dataset/soulx_duplug_eval/reports/"
    "benchmark_leakage_denylist_v2/benchmark_identity_manifest.jsonl"
)
FROZEN_LEAKAGE_CONFIG = PROJECT_ROOT / "configs/benchmark_audio_leakage_frozen_v2_2.json"
TOKENIZER_DIR = PROJECT_ROOT / "pretrained_models/Qwen3-0.6B-expand_vocab_v2"
PARAFORMER_MODEL = PROJECT_ROOT / "pretrained_models/paraformer-zh"
GLM_MODEL = PROJECT_ROOT / "pretrained_models/glm-4-voice-tokenizer"
UPSTREAM_DIR = PROJECT_ROOT / "third_party/SoulX-Duplug-upstream"
UPSTREAM_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
ENV_FILE = PROJECT_ROOT / ".env"
QWEN_MODEL = "qwen/qwen3-235b-a22b-2507"
QWEN_ROUTE = "direct-no-proxy-v2"
MINIMUM_FREE_BYTES = 20 * 1024**3
RUN_VARIANT_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
SUPPORTED_QWEN_ROUTES = {
    "direct-no-proxy-v2",
    "environment-proxy-aware-v2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def update(state: dict[str, Any], manifest: Path, status: str, stage: str, **extra: Any) -> None:
    state.update(extra)
    state["status"] = status
    state["stage"] = stage
    state["updated_at_utc"] = utc_now()
    atomic_json(manifest, state)


def run_logged(
    state: dict[str, Any],
    manifest: Path,
    stage: str,
    command: list[str],
    log_path: Path,
    *,
    cwd: Path = PROJECT_ROOT,
) -> None:
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite stage log: {log_path}")
    update(
        state,
        manifest,
        "running",
        stage,
        current_log=str(log_path),
        current_command=[Path(command[0]).name, *command[1:]],
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage {stage} failed with return code {completed.returncode}; log={log_path}"
        )
    state.setdefault("completed_stages", []).append(
        {"stage": stage, "completed_at_utc": utc_now(), "log": str(log_path)}
    )
    update(state, manifest, "running", f"{stage}_complete")


def wait_for_download(state: dict[str, Any], manifest: Path, download_manifest: Path) -> dict[str, Any]:
    update(state, manifest, "waiting", "verified_download")
    while True:
        if download_manifest.exists():
            value = load_json(download_manifest)
            status = value.get("status")
            update(
                state,
                manifest,
                "waiting" if status not in {"complete", "already_complete"} else "running",
                "verified_download",
                download_status=status,
                download_updated_at_utc=value.get("updated_at_utc"),
                download_progress_bytes=value.get("downloaded_bytes"),
            )
            if status in {"complete", "already_complete"}:
                return value
            if status == "failed":
                raise RuntimeError("download manifest reports failure")
        time.sleep(15)


def tar_source_ids(path: Path) -> list[str]:
    values: list[str] = []
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".wav"):
                values.append(PurePosixPath(member.name).stem)
    if len(values) != len(set(values)):
        raise RuntimeError("audio archive contains duplicate source IDs")
    return sorted(values)


def source_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def validate_run_variant(value: str | None) -> str | None:
    if value is None:
        return None
    if not RUN_VARIANT_PATTERN.fullmatch(value):
        raise ValueError(
            "run variant must contain only lowercase letters, digits, and single underscores"
        )
    return value


def validate_qwen_route(value: str) -> str:
    if value not in SUPPORTED_QWEN_ROUTES:
        raise ValueError(f"unsupported Qwen route: {value}")
    return value


def parse_route_count_arguments(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        route, separator, raw_count = value.rpartition("=")
        if not separator or route not in SUPPORTED_QWEN_ROUTES:
            raise ValueError(f"invalid expected result route count: {value}")
        if route in counts:
            raise ValueError(f"duplicate expected result route: {route}")
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(f"invalid expected result route count: {value}") from None
        if count < 0:
            raise ValueError(f"negative expected result route count: {value}")
        counts[route] = count
    return counts


def variant_output_name(name: str, run_variant: str | None) -> str:
    """Keep legacy names unchanged; insert a versioned variant when requested."""

    run_variant = validate_run_variant(run_variant)
    if run_variant is None:
        return name
    marker_index = len(name)
    for marker in ("_v2_2", "_v1"):
        candidate = name.find(marker)
        if candidate >= 0:
            marker_index = min(marker_index, candidate)
    if marker_index < len(name):
        return f"{name[:marker_index]}_{run_variant}{name[marker_index:]}"
    path = Path(name)
    return f"{path.stem}_{run_variant}_v1{path.suffix}"


def benchmark_identifiers(path: Path) -> tuple[int, set[str]]:
    identifiers: set[str] = set()
    record_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_count += 1
            row = json.loads(line)
            for key in ("record_id", "sample_key", "member", "zip", "numeric_sample_id"):
                value = row.get(key)
                if value is None:
                    continue
                text = str(value)
                identifiers.add(text)
                identifiers.add(PurePosixPath(text).name)
                identifiers.add(PurePosixPath(text).stem)
    return record_count, identifiers


def write_gate_a(
    contract: dict[str, Any],
    archive: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Gate A report: {output}")
    source_ids = tar_source_ids(archive)
    expected = contract["expected"]
    membership_checks = {
        "source_count": len(source_ids) == expected["source_count"],
        "first_source_id": source_ids[0] == expected["first_source_id"],
        "last_source_id": source_ids[-1] == expected["last_source_id"],
        "source_ids_sha256": source_ids_sha256(source_ids) == expected["source_ids_sha256"],
    }
    if not all(membership_checks.values()):
        raise RuntimeError(f"official tar membership contract failed: {membership_checks}")
    record_count, identifiers = benchmark_identifiers(BENCHMARK_RECORDS)
    collisions = sorted(set(source_ids) & identifiers)
    report = {
        "schema_version": 1,
        "profile": "duplexconv-benchmark-metadata-identity-gate-a-v1",
        "completed_at_utc": utc_now(),
        "candidate_definition": (
            "Exact source IDs shared by the deterministic sorted-metadata shard slice "
            "and the verified official tar membership"
        ),
        "candidate_source_count": len(source_ids),
        "candidate_first_source_id": source_ids[0],
        "candidate_last_source_id": source_ids[-1],
        "candidate_source_ids_sha256": source_ids_sha256(source_ids),
        "official_tar_membership_exact_match": True,
        "official_tar_membership_checks": membership_checks,
        "benchmark_record_count": record_count,
        "benchmark_identifier_string_count": len(identifiers),
        "compared_benchmark_fields": [
            "record_id",
            "sample_key",
            "member",
            "zip",
            "numeric_sample_id",
            "path basename",
            "path stem",
        ],
        "exact_source_identifier_collision_count": len(collisions),
        "collisions": collisions,
        "benchmark_identity_manifest_sha256": sha256_file(BENCHMARK_RECORDS),
        "gate_passed": not collisions,
    }
    atomic_json(output, report)
    if collisions:
        raise RuntimeError(f"Gate A found benchmark identifier collisions: {collisions}")
    return report


def verify_checksums(path: Path) -> None:
    completed = subprocess.run(
        ["sha256sum", "-c", "--quiet", "checksums.sha256"],
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"checksum verification failed for {path}: {completed.stdout}")


def require_fresh(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")


def run_pipeline(
    contract_path: Path,
    download_manifest: Path,
    *,
    archive_path: Path | None = None,
    run_variant: str | None = None,
    excluded_source_id: str | None = None,
    qwen_route: str = QWEN_ROUTE,
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    download_manifest = download_manifest.absolute()
    contract = load_json(contract_path)
    run_variant = validate_run_variant(run_variant)
    qwen_route = validate_qwen_route(qwen_route)
    if run_variant is None and (archive_path is not None or excluded_source_id is not None):
        raise ValueError("archive override and source exclusion require an explicit run variant")
    if excluded_source_id is not None:
        contract_exclusions = contract.get("excluded_source_ids")
        if contract_exclusions != [excluded_source_id]:
            raise ValueError(
                "contract excluded_source_ids must exactly match --excluded-source-id"
            )
    shard_number = int(contract["shard_number"])
    shard = f"edu{shard_number:04d}"
    shard_title = f"Edu_{shard_number:04d}"
    tag = f"{shard}_{run_variant}_v1" if run_variant else f"{shard}_v1"
    dataset_version = contract["dataset_version"]
    index_prefix = f"duplexconv_{tag}"

    archive = (
        archive_path.resolve(strict=True)
        if archive_path is not None
        else DATA_ROOT / "raw/audio" / contract["archive_name"]
    )
    report_dir = DATA_ROOT / "reports/expansion_v1" / shard_title
    report_artifact = lambda name: report_dir / variant_output_name(name, run_variant)
    orchestration_manifest = report_artifact("full_pipeline_manifest_v1.json")
    gate_a_report = report_artifact("gate_a_metadata_v1.json")
    source_scan = DATA_ROOT / "work" / f"source_scan_{tag}"
    leakage = report_artifact("leakage_gate_v2_2")
    requests = DATA_ROOT / "work" / f"state_label_requests_{tag}"
    openrouter_cache = DATA_ROOT / "cache" / f"openrouter_state_labels_{tag}"
    state_labels = DATA_ROOT / "processed" / f"state_labels_{tag}"
    target_audio = DATA_ROOT / "processed" / f"target_audio_{tag}"
    paraformer = DATA_ROOT / "processed" / f"paraformer_{tag}"
    paraformer_cache = DATA_ROOT / "cache" / f"paraformer_{tag}"
    rendered = DATA_ROOT / "processed" / f"paraformer_rendered_{tag}"
    timelines = DATA_ROOT / "processed" / f"timelines_{tag}"
    glm = DATA_ROOT / "processed" / f"glm_audio_tokens_{tag}"
    glm_cache = DATA_ROOT / "cache" / f"glm_audio_tokens_{tag}"
    model_ready_name = (
        f"{shard}_{run_variant}_stage3_zh_v1"
        if run_variant
        else f"{shard}_stage3_zh_v1"
    )
    model_ready = DATA_ROOT / "model_ready" / model_ready_name
    model_validation = report_artifact("model_ready_validation_v1.json")
    gate_d_work = DATA_ROOT / "work" / f"gate_d_{tag}"
    gate_d_selection = gate_d_work / "selection.jsonl"
    gate_d_tar = gate_d_work / "candidate_model_ready_views.tar"
    gate_d_build_manifest = gate_d_work / "build_manifest.json"
    gate_d_output = report_artifact("leakage_gate_d_v1")
    gate_d_closure = report_artifact("gate_d_closure_v1.json")

    require_fresh(
        [
            orchestration_manifest,
            gate_a_report,
            source_scan,
            leakage,
            requests,
            openrouter_cache,
            state_labels,
            target_audio,
            paraformer,
            paraformer_cache,
            rendered,
            timelines,
            glm,
            glm_cache,
            model_ready,
            model_validation,
            gate_d_work,
            gate_d_output,
            gate_d_closure,
        ]
    )
    state: dict[str, Any] = {
        "schema_version": 1,
        "profile": "duplexconv-frozen-expansion-full-pipeline-v1",
        "status": "starting",
        "stage": "preflight",
        "started_at_utc": utc_now(),
        "approved_scope": f"{shard_title} data construction through final frozen Gate D only",
        "forbidden_actions": ["training", "checkpoint_evaluation", "benchmark_rule_changes"],
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "download_manifest": str(download_manifest),
        "run_variant": run_variant,
        "archive_override": str(archive) if archive_path is not None else None,
        "excluded_source_id": excluded_source_id,
        "qwen_model": QWEN_MODEL,
        "qwen_route": qwen_route,
        "qwen_workers": 4,
        "daily_budget_cap_usd": 10.0,
        "completed_stages": [],
    }
    atomic_json(orchestration_manifest, state)
    try:
        download = wait_for_download(state, orchestration_manifest, download_manifest)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        expected_archive_bytes = contract.get(
            "input_archive_bytes", contract.get("official_archive_bytes")
        )
        expected_archive_sha256 = contract.get(
            "input_archive_sha256", contract.get("official_archive_sha256")
        )
        if expected_archive_bytes is None or expected_archive_sha256 is None:
            raise RuntimeError("contract does not define the selected input archive identity")
        if archive.stat().st_size != int(expected_archive_bytes):
            raise RuntimeError("downloaded archive byte count differs from contract")
        if sha256_file(archive) != expected_archive_sha256:
            raise RuntimeError("downloaded archive SHA-256 differs from contract")
        state["download"] = {
            "status": download["status"],
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": expected_archive_sha256,
            "network_route": download.get("network_route"),
        }

        update(state, orchestration_manifest, "running", "gate_a_metadata_identity")
        state["gate_a"] = write_gate_a(contract, archive, gate_a_report)
        state["completed_stages"].append(
            {"stage": "gate_a_metadata_identity", "completed_at_utc": utc_now(), "report": str(gate_a_report)}
        )

        run_logged(
            state,
            orchestration_manifest,
            "source_scan",
            [
                str(PYTHON),
                "scripts/scan_duplexconv_source.py",
                "--audio-archive",
                str(archive),
                "--metadata-archive",
                str(METADATA_ARCHIVE),
                "--output-dir",
                str(source_scan),
                "--contract-json",
                str(contract_path),
            ],
            report_artifact("source_scan.log"),
        )
        scan_summary = load_json(source_scan / "summary.json")
        if not scan_summary.get("gate_3_source_contract_passed"):
            raise RuntimeError("source contract gate did not pass")
        if scan_summary["sources"].get("quarantined") != 0:
            raise RuntimeError("source scan found a structural quarantine outside automatic frozen repair")

        run_logged(
            state,
            orchestration_manifest,
            "frozen_leakage_gate_b_c",
            [
                str(PYTHON),
                "scripts/run_audio_leakage_gate_v2_2.py",
                "score-tar",
                "--candidate-tar",
                str(archive),
                "--benchmark-records",
                str(BENCHMARK_RECORDS),
                "--frozen-config",
                str(FROZEN_LEAKAGE_CONFIG),
                "--output-dir",
                str(leakage),
            ],
            report_artifact("leakage.log"),
        )
        leakage_manifest = load_json(leakage / "run_manifest.json")
        if not leakage_manifest.get("gate_passed"):
            raise RuntimeError(
                "frozen Gate B/C found benchmark leakage; stop before paid Qwen calls"
            )

        run_logged(
            state,
            orchestration_manifest,
            "prepare_qwen_requests",
            [
                str(PYTHON),
                "scripts/prepare_state_label_requests.py",
                "--scan-dir",
                str(source_scan),
                "--output-dir",
                str(requests),
                "--tokenizer-path",
                str(TOKENIZER_DIR),
                "--calibration-per-state",
                "10",
            ],
            report_artifact("prepare_qwen_requests.log"),
        )
        request_summary = load_json(requests / "summary.json")
        expected_missing = int(scan_summary["events"]["state_distribution"]["missing"])
        if int(request_summary["full_target_event_count"]) != expected_missing:
            raise RuntimeError("Qwen request/event closure differs from source-scan missing states")

        run_logged(
            state,
            orchestration_manifest,
            "qwen_connectivity",
            [
                str(PYTHON),
                "scripts/run_openrouter_labeling_with_manifest.py",
                "--request-file",
                str(requests / "connectivity_request.json"),
                "--cache-dir",
                str(openrouter_cache),
                "--result-file",
                str(requests / "connectivity_results.jsonl"),
                "--env-file",
                str(ENV_FILE),
                "--manifest",
                str(requests / "connectivity_labeling_run_manifest.json"),
                "--workers",
                "1",
                "--network-route-policy",
                qwen_route,
            ],
            report_artifact("connectivity_labeling.log"),
        )

        run_logged(
            state,
            orchestration_manifest,
            "qwen_calibration",
            [
                str(PYTHON),
                "scripts/run_openrouter_labeling_with_manifest.py",
                "--request-file",
                str(requests / "calibration_requests.jsonl"),
                "--cache-dir",
                str(openrouter_cache),
                "--result-file",
                str(requests / "calibration_results.jsonl"),
                "--env-file",
                str(ENV_FILE),
                "--manifest",
                str(requests / "calibration_labeling_run_manifest.json"),
                "--workers",
                "4",
                "--network-route-policy",
                qwen_route,
            ],
            report_artifact("calibration_labeling.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "qwen_calibration_audit",
            [
                str(PYTHON),
                "scripts/audit_openrouter_state_labels.py",
                "--request-file",
                str(requests / "calibration_requests.jsonl"),
                "--result-file",
                str(requests / "calibration_results.jsonl"),
                "--run-manifest",
                str(requests / "calibration_labeling_run_manifest.json"),
                "--output",
                str(requests / "calibration_audit.json"),
                "--expected-kind",
                "calibration",
                "--expected-model",
                QWEN_MODEL,
                "--expected-route",
                qwen_route,
                "--env-file",
                str(ENV_FILE),
                "--calibration-answer-key",
                str(requests / "calibration_answer_key.json"),
            ],
            report_artifact("calibration_audit.log"),
        )
        calibration_audit = load_json(requests / "calibration_audit.json")
        if not calibration_audit.get("mechanical_gate_passed") or not calibration_audit.get(
            "proceed_to_full_run"
        ):
            raise RuntimeError("Qwen calibration mechanical gate did not pass")

        confirmation = str(request_summary["full_run_confirmation"])
        run_logged(
            state,
            orchestration_manifest,
            "qwen_full_labeling",
            [
                str(PYTHON),
                "scripts/run_openrouter_labeling_with_manifest.py",
                "--request-file",
                str(requests / "full_requests.jsonl"),
                "--cache-dir",
                str(openrouter_cache),
                "--result-file",
                str(requests / "full_results.jsonl"),
                "--env-file",
                str(ENV_FILE),
                "--manifest",
                str(requests / "full_labeling_run_manifest.json"),
                "--workers",
                "4",
                "--network-route-policy",
                qwen_route,
                "--full-run-confirmation",
                confirmation,
            ],
            report_artifact("full_labeling.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "qwen_full_audit",
            [
                str(PYTHON),
                "scripts/audit_openrouter_state_labels.py",
                "--request-file",
                str(requests / "full_requests.jsonl"),
                "--result-file",
                str(requests / "full_results.jsonl"),
                "--run-manifest",
                str(requests / "full_labeling_run_manifest.json"),
                "--output",
                str(requests / "full_labeling_audit.json"),
                "--expected-kind",
                "full",
                "--expected-model",
                QWEN_MODEL,
                "--expected-route",
                qwen_route,
                "--env-file",
                str(ENV_FILE),
            ],
            report_artifact("full_labeling_audit.log"),
        )
        full_audit = load_json(requests / "full_labeling_audit.json")
        if not full_audit.get("mechanical_gate_passed") or not full_audit.get(
            "proceed_to_state_finalization"
        ):
            raise RuntimeError("full Qwen labeling mechanical gate did not pass")

        run_logged(
            state,
            orchestration_manifest,
            "state_finalization",
            [
                str(PYTHON),
                "scripts/finalize_state_labels.py",
                "--scan-dir",
                str(source_scan),
                "--request-dir",
                str(requests),
                "--output-dir",
                str(state_labels),
            ],
            report_artifact("state_finalization.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "target_audio",
            [
                str(PYTHON),
                "scripts/extract_target_audio.py",
                "--audio-archive",
                str(archive),
                "--scan-dir",
                str(source_scan),
                "--output-dir",
                str(target_audio),
                "--all",
            ],
            report_artifact("target_audio.log"),
        )
        verify_checksums(target_audio)

        run_logged(
            state,
            orchestration_manifest,
            "paraformer",
            [
                str(PYTHON),
                "scripts/run_paraformer.py",
                "--model-dir",
                str(PARAFORMER_MODEL),
                "--audio-dir",
                str(target_audio),
                "--scan-dir",
                str(source_scan),
                "--output-dir",
                str(paraformer),
                "--cache-dir",
                str(paraformer_cache),
                "--device",
                "cuda:0",
                "--all",
            ],
            report_artifact("paraformer.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "render_paraformer",
            [
                str(PYTHON),
                "scripts/render_paraformer_text.py",
                "--input-dir",
                str(paraformer),
                "--output-dir",
                str(rendered),
            ],
            report_artifact("render_paraformer.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "timelines",
            [
                str(PYTHON),
                "scripts/build_timelines.py",
                "--scan-dir",
                str(source_scan),
                "--state-dir",
                str(state_labels),
                "--asr-dir",
                str(rendered),
                "--output-dir",
                str(timelines),
            ],
            report_artifact("timelines.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "glm_audio_tokens",
            [
                str(PYTHON),
                "scripts/extract_glm_audio_tokens.py",
                "--upstream-dir",
                str(UPSTREAM_DIR),
                "--model-dir",
                str(GLM_MODEL),
                "--audio-dir",
                str(target_audio),
                "--timeline-dir",
                str(timelines),
                "--output-dir",
                str(glm),
                "--cache-dir",
                str(glm_cache),
                "--device",
                "cuda:0",
                "--batch-size",
                "16",
                "--all",
            ],
            report_artifact("glm_audio_tokens.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "model_ready_export",
            [
                str(PYTHON),
                "scripts/export_model_ready.py",
                "--timeline-dir",
                str(timelines),
                "--glm-dir",
                str(glm),
                "--tokenizer-dir",
                str(TOKENIZER_DIR),
                "--output-dir",
                str(model_ready),
                "--upstream-commit",
                UPSTREAM_COMMIT,
                "--max-token-length",
                "1500",
                "--dataset-version",
                dataset_version,
                "--index-prefix",
                index_prefix,
            ],
            report_artifact("model_ready.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "model_ready_validation",
            [
                str(PYTHON),
                "scripts/validate_model_ready.py",
                "--model-ready-dir",
                str(model_ready),
                "--tokenizer-dir",
                str(TOKENIZER_DIR),
                "--upstream-dir",
                str(UPSTREAM_DIR),
                "--report-path",
                str(model_validation),
                "--random-sample-count",
                "20",
            ],
            report_artifact("model_ready_validation.log"),
        )
        validation = load_json(model_validation)
        if validation.get("status") != "passed" or not validation.get(
            "global_view_and_chunk_closure_passed"
        ):
            raise RuntimeError("model-ready validation did not pass")

        target_audio_bytes = sum(
            path.stat().st_size for path in target_audio.rglob("*.wav") if path.is_file()
        )
        available = shutil.disk_usage(DATA_ROOT).free
        if available < target_audio_bytes + MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "Gate D disk gate failed: "
                f"available={available}, candidate_upper_bound={target_audio_bytes}, "
                f"minimum_free={MINIMUM_FREE_BYTES}"
            )
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_build",
            [
                str(PYTHON),
                "scripts/build_model_ready_gate_d_tar.py",
                "--model-ready-dir",
                str(model_ready),
                "--target-audio-dir",
                str(target_audio),
                "--selection-list",
                str(gate_d_selection),
                "--output-tar",
                str(gate_d_tar),
                "--manifest",
                str(gate_d_build_manifest),
            ],
            report_artifact("gate_d_build.log"),
        )
        gate_d_build = load_json(gate_d_build_manifest)
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_score",
            [
                str(PYTHON),
                "scripts/run_audio_leakage_gate_v2_2.py",
                "score-tar",
                "--candidate-tar",
                str(gate_d_tar),
                "--benchmark-records",
                str(BENCHMARK_RECORDS),
                "--frozen-config",
                str(FROZEN_LEAKAGE_CONFIG),
                "--output-dir",
                str(gate_d_output),
            ],
            report_artifact("gate_d_scorer.log"),
        )
        gate_d_manifest = load_json(gate_d_output / "run_manifest.json")
        if not gate_d_manifest.get("gate_passed"):
            raise RuntimeError("final frozen Gate D did not pass")
        closure_command = [
            str(PYTHON),
            "scripts/audit_model_ready_gate_d_closure.py",
            "--model-ready-dir",
            str(model_ready),
            "--model-ready-validation",
            str(model_validation),
            "--target-audio-manifest",
            str(target_audio / "audio_manifest.jsonl"),
            "--selection-list",
            str(gate_d_selection),
            "--candidate-tar",
            str(gate_d_tar),
            "--gate-output-dir",
            str(gate_d_output),
            "--benchmark-records",
            str(BENCHMARK_RECORDS),
            "--frozen-config",
            str(FROZEN_LEAKAGE_CONFIG),
            "--expected-view-count",
            str(gate_d_build["model_ready_view_count"]),
            "--expected-source-count",
            str(gate_d_build["source_id_count"]),
        ]
        if excluded_source_id is not None:
            closure_command.extend(["--excluded-source-id", excluded_source_id])
        closure_command.extend(["--output", str(gate_d_closure)])
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_closure_audit",
            closure_command,
            report_artifact("gate_d_closure_audit.log"),
        )

        for artifact in (
            state_labels,
            target_audio,
            paraformer,
            rendered,
            timelines,
            glm,
            model_ready,
        ):
            verify_checksums(artifact)
        closure = load_json(gate_d_closure)
        state["result"] = {
            "source_count": scan_summary["sources"]["total"],
            "target_view_count_before_downstream_quarantine": scan_summary["views"][
                "structurally_usable"
            ],
            "target_view_hours": scan_summary["views"]["total_duration_hours"],
            "missing_state_event_count": expected_missing,
            "model_ready_row_count": validation["row_count"],
            "model_ready_view_count": validation["source_view_count"],
            "model_ready_exported_chunk_count": validation["exported_chunk_count"],
            "model_ready_dir": str(model_ready),
            "gate_d_closure": str(gate_d_closure),
            "gate_d_closure_sha256": sha256_file(gate_d_closure),
            "gate_d_passed": closure.get("gate_passed") is True,
            "full_qwen_accepted_cost_usd": full_audit["accepted_response_usage"]["cost_usd"],
        }
        update(
            state,
            orchestration_manifest,
            "complete",
            "final_frozen_gate_d_complete",
            completed_at_utc=utc_now(),
            current_log=None,
            current_command=None,
        )
        return state
    except BaseException as exc:
        update(
            state,
            orchestration_manifest,
            "failed",
            state.get("stage", "unknown"),
            failed_at_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def validate_resume_after_qwen_full(
    state: dict[str, Any],
    failed_snapshot: dict[str, Any],
    failed_labeling: dict[str, Any],
    recovery: dict[str, Any],
    *,
    contract_sha256: str,
    request_file_sha256: str,
    result_file: Path,
    expected_route_counts: dict[str, int],
) -> None:
    expected_stages = [
        "gate_a_metadata_identity",
        "source_scan",
        "frozen_leakage_gate_b_c",
        "prepare_qwen_requests",
        "qwen_connectivity",
        "qwen_calibration",
        "qwen_calibration_audit",
    ]
    if state != failed_snapshot:
        raise RuntimeError("live failed pipeline manifest differs from immutable snapshot")
    if state.get("status") != "failed" or state.get("stage") != "qwen_full_labeling":
        raise RuntimeError("pipeline is not a failed qwen_full_labeling run")
    if state.get("contract_sha256") != contract_sha256:
        raise RuntimeError("resume contract hash differs from failed pipeline")
    completed = [item.get("stage") for item in state.get("completed_stages", [])]
    if completed != expected_stages:
        raise RuntimeError(f"unexpected completed-stage prefix for resume: {completed}")
    if failed_labeling.get("status") != "failed":
        raise RuntimeError("original full-labeling manifest is not failed")
    if failed_labeling.get("request_file_sha256") != request_file_sha256:
        raise RuntimeError("failed labeling request hash differs from frozen request file")
    if recovery.get("status") != "complete":
        raise RuntimeError("cache-safe recovery manifest is not complete")
    if recovery.get("request_file_sha256") != request_file_sha256:
        raise RuntimeError("recovery request hash differs from frozen request file")
    if Path(recovery.get("result_file", "")).resolve() != result_file.resolve():
        raise RuntimeError("recovery result path differs from frozen full result path")
    if not result_file.is_file():
        raise RuntimeError("cache-safe recovery did not publish the full result file")
    if recovery.get("result_file_sha256") != sha256_file(result_file):
        raise RuntimeError("recovery result hash differs from the published result file")
    summary = recovery.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("recovery manifest has no summary")
    if summary.get("request_count") != failed_labeling.get("request_count"):
        raise RuntimeError("recovery request count differs from the failed full run")
    if summary.get("result_count") != failed_labeling.get("request_count"):
        raise RuntimeError("recovery did not close every frozen full request")
    if summary.get("target_event_count") != failed_labeling.get("target_event_count"):
        raise RuntimeError("recovery target-event count differs from the failed full run")
    if sum(expected_route_counts.values()) != failed_labeling.get("request_count"):
        raise RuntimeError("expected recovery route counts do not close the request set")
    if summary.get("result_network_route_policy_counts") != expected_route_counts:
        raise RuntimeError("recovery result route distribution differs from approval")


def resume_pipeline_after_qwen_full(
    contract_path: Path,
    recovery_manifest_path: Path,
    failed_pipeline_snapshot_path: Path,
    *,
    archive_path: Path,
    run_variant: str,
    excluded_source_id: str,
    expected_route_counts: dict[str, int],
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    recovery_manifest_path = recovery_manifest_path.resolve(strict=True)
    failed_pipeline_snapshot_path = failed_pipeline_snapshot_path.resolve(strict=True)
    archive = archive_path.resolve(strict=True)
    run_variant = validate_run_variant(run_variant)
    if run_variant is None:
        raise ValueError("resume requires an explicit run variant")
    contract = load_json(contract_path)
    if contract.get("excluded_source_ids") != [excluded_source_id]:
        raise ValueError("resume exclusion differs from the frozen contract")

    shard_number = int(contract["shard_number"])
    shard = f"edu{shard_number:04d}"
    shard_title = f"Edu_{shard_number:04d}"
    tag = f"{shard}_{run_variant}_v1"
    dataset_version = contract["dataset_version"]
    index_prefix = f"duplexconv_{tag}"
    report_dir = DATA_ROOT / "reports/expansion_v1" / shard_title
    report_artifact = lambda name: report_dir / variant_output_name(name, run_variant)

    orchestration_manifest = report_artifact("full_pipeline_manifest_v1.json")
    source_scan = DATA_ROOT / "work" / f"source_scan_{tag}"
    leakage = report_artifact("leakage_gate_v2_2")
    requests = DATA_ROOT / "work" / f"state_label_requests_{tag}"
    state_labels = DATA_ROOT / "processed" / f"state_labels_{tag}"
    target_audio = DATA_ROOT / "processed" / f"target_audio_{tag}"
    paraformer = DATA_ROOT / "processed" / f"paraformer_{tag}"
    paraformer_cache = DATA_ROOT / "cache" / f"paraformer_{tag}"
    rendered = DATA_ROOT / "processed" / f"paraformer_rendered_{tag}"
    timelines = DATA_ROOT / "processed" / f"timelines_{tag}"
    glm = DATA_ROOT / "processed" / f"glm_audio_tokens_{tag}"
    glm_cache = DATA_ROOT / "cache" / f"glm_audio_tokens_{tag}"
    model_ready = DATA_ROOT / "model_ready" / f"{shard}_{run_variant}_stage3_zh_v1"
    model_validation = report_artifact("model_ready_validation_v1.json")
    gate_d_work = DATA_ROOT / "work" / f"gate_d_{tag}"
    gate_d_selection = gate_d_work / "selection.jsonl"
    gate_d_tar = gate_d_work / "candidate_model_ready_views.tar"
    gate_d_build_manifest = gate_d_work / "build_manifest.json"
    gate_d_output = report_artifact("leakage_gate_d_v1")
    gate_d_closure = report_artifact("gate_d_closure_v1.json")
    failed_labeling_manifest = requests / "full_labeling_run_manifest.json"
    result_file = requests / "full_results.jsonl"
    full_audit_path = requests / "full_labeling_audit.json"

    state = load_json(orchestration_manifest)
    failed_snapshot = load_json(failed_pipeline_snapshot_path)
    failed_labeling = load_json(failed_labeling_manifest)
    recovery = load_json(recovery_manifest_path)
    request_file = requests / "full_requests.jsonl"
    request_hash = sha256_file(request_file)
    validate_resume_after_qwen_full(
        state,
        failed_snapshot,
        failed_labeling,
        recovery,
        contract_sha256=sha256_file(contract_path),
        request_file_sha256=request_hash,
        result_file=result_file,
        expected_route_counts=expected_route_counts,
    )
    expected_archive_bytes = int(contract["input_archive_bytes"])
    expected_archive_sha256 = contract["input_archive_sha256"]
    if archive.stat().st_size != expected_archive_bytes:
        raise RuntimeError("resume input archive byte count differs from contract")
    if sha256_file(archive) != expected_archive_sha256:
        raise RuntimeError("resume input archive hash differs from contract")

    scan_summary = load_json(source_scan / "summary.json")
    leakage_manifest = load_json(leakage / "run_manifest.json")
    calibration_audit = load_json(requests / "calibration_audit.json")
    request_summary = load_json(requests / "summary.json")
    expected_missing = int(scan_summary["events"]["state_distribution"]["missing"])
    if not scan_summary.get("gate_3_source_contract_passed"):
        raise RuntimeError("resume source contract is not passed")
    if scan_summary["sources"].get("quarantined") != 0:
        raise RuntimeError("resume source scan contains structural quarantine")
    if not leakage_manifest.get("gate_passed"):
        raise RuntimeError("resume frozen Gate B/C is not passed")
    if not calibration_audit.get("mechanical_gate_passed") or not calibration_audit.get(
        "proceed_to_full_run"
    ):
        raise RuntimeError("resume calibration gate is not passed")
    if int(request_summary["full_target_event_count"]) != expected_missing:
        raise RuntimeError("resume request/event closure differs from source scan")

    require_fresh(
        [
            full_audit_path,
            state_labels,
            target_audio,
            paraformer,
            paraformer_cache,
            rendered,
            timelines,
            glm,
            glm_cache,
            model_ready,
            model_validation,
            gate_d_work,
            gate_d_output,
            gate_d_closure,
        ]
    )
    state["resume_after_qwen_full"] = {
        "failed_pipeline_snapshot": str(failed_pipeline_snapshot_path),
        "failed_pipeline_snapshot_sha256": sha256_file(failed_pipeline_snapshot_path),
        "failed_labeling_manifest": str(failed_labeling_manifest),
        "failed_labeling_manifest_sha256": sha256_file(failed_labeling_manifest),
        "recovery_manifest": str(recovery_manifest_path),
        "recovery_manifest_sha256": sha256_file(recovery_manifest_path),
        "recovery_result_file": str(result_file),
        "recovery_result_file_sha256": sha256_file(result_file),
        "request_file_sha256": request_hash,
        "approved_result_network_route_policy_counts": expected_route_counts,
        "policy": "Reuse exact signature-validated caches and send only missing requests; do not select outputs or change model, prompt, route, or request set.",
    }
    state.setdefault("completed_stages", []).append(
        {
            "stage": "qwen_full_labeling_cache_safe_recovery",
            "completed_at_utc": recovery.get("completed_at_utc"),
            "manifest": str(recovery_manifest_path),
        }
    )
    update(
        state,
        orchestration_manifest,
        "running",
        "qwen_full_recovery_validated",
        failed_at_utc=state.get("failed_at_utc"),
        error_type=None,
        error=None,
    )

    try:
        audit_command = [
            str(PYTHON),
            "scripts/audit_openrouter_state_labels.py",
            "--request-file",
            str(request_file),
            "--result-file",
            str(result_file),
            "--run-manifest",
            str(recovery_manifest_path),
            "--output",
            str(full_audit_path),
            "--expected-kind",
            "full",
            "--expected-model",
            QWEN_MODEL,
            "--expected-route",
            str(recovery["network_route_policy"]),
            "--env-file",
            str(ENV_FILE),
        ]
        for route, count in sorted(expected_route_counts.items()):
            audit_command.extend(["--expected-route-count", f"{route}={count}"])
        run_logged(
            state,
            orchestration_manifest,
            "qwen_full_audit",
            audit_command,
            report_artifact("full_labeling_audit.log"),
        )
        full_audit = load_json(full_audit_path)
        if not full_audit.get("mechanical_gate_passed") or not full_audit.get(
            "proceed_to_state_finalization"
        ):
            raise RuntimeError("recovered full Qwen labeling mechanical gate did not pass")

        run_logged(
            state,
            orchestration_manifest,
            "state_finalization",
            [
                str(PYTHON),
                "scripts/finalize_state_labels.py",
                "--scan-dir",
                str(source_scan),
                "--request-dir",
                str(requests),
                "--output-dir",
                str(state_labels),
            ],
            report_artifact("state_finalization.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "target_audio",
            [
                str(PYTHON),
                "scripts/extract_target_audio.py",
                "--audio-archive",
                str(archive),
                "--scan-dir",
                str(source_scan),
                "--output-dir",
                str(target_audio),
                "--all",
            ],
            report_artifact("target_audio.log"),
        )
        verify_checksums(target_audio)
        run_logged(
            state,
            orchestration_manifest,
            "paraformer",
            [
                str(PYTHON),
                "scripts/run_paraformer.py",
                "--model-dir",
                str(PARAFORMER_MODEL),
                "--audio-dir",
                str(target_audio),
                "--scan-dir",
                str(source_scan),
                "--output-dir",
                str(paraformer),
                "--cache-dir",
                str(paraformer_cache),
                "--device",
                "cuda:0",
                "--all",
            ],
            report_artifact("paraformer.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "render_paraformer",
            [
                str(PYTHON),
                "scripts/render_paraformer_text.py",
                "--input-dir",
                str(paraformer),
                "--output-dir",
                str(rendered),
            ],
            report_artifact("render_paraformer.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "timelines",
            [
                str(PYTHON),
                "scripts/build_timelines.py",
                "--scan-dir",
                str(source_scan),
                "--state-dir",
                str(state_labels),
                "--asr-dir",
                str(rendered),
                "--output-dir",
                str(timelines),
            ],
            report_artifact("timelines.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "glm_audio_tokens",
            [
                str(PYTHON),
                "scripts/extract_glm_audio_tokens.py",
                "--upstream-dir",
                str(UPSTREAM_DIR),
                "--model-dir",
                str(GLM_MODEL),
                "--audio-dir",
                str(target_audio),
                "--timeline-dir",
                str(timelines),
                "--output-dir",
                str(glm),
                "--cache-dir",
                str(glm_cache),
                "--device",
                "cuda:0",
                "--batch-size",
                "16",
                "--all",
            ],
            report_artifact("glm_audio_tokens.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "model_ready_export",
            [
                str(PYTHON),
                "scripts/export_model_ready.py",
                "--timeline-dir",
                str(timelines),
                "--glm-dir",
                str(glm),
                "--tokenizer-dir",
                str(TOKENIZER_DIR),
                "--output-dir",
                str(model_ready),
                "--upstream-commit",
                UPSTREAM_COMMIT,
                "--max-token-length",
                "1500",
                "--dataset-version",
                dataset_version,
                "--index-prefix",
                index_prefix,
            ],
            report_artifact("model_ready.log"),
        )
        run_logged(
            state,
            orchestration_manifest,
            "model_ready_validation",
            [
                str(PYTHON),
                "scripts/validate_model_ready.py",
                "--model-ready-dir",
                str(model_ready),
                "--tokenizer-dir",
                str(TOKENIZER_DIR),
                "--upstream-dir",
                str(UPSTREAM_DIR),
                "--report-path",
                str(model_validation),
                "--random-sample-count",
                "20",
            ],
            report_artifact("model_ready_validation.log"),
        )
        validation = load_json(model_validation)
        if validation.get("status") != "passed" or not validation.get(
            "global_view_and_chunk_closure_passed"
        ):
            raise RuntimeError("model-ready validation did not pass")

        target_audio_bytes = sum(
            path.stat().st_size for path in target_audio.rglob("*.wav") if path.is_file()
        )
        available = shutil.disk_usage(DATA_ROOT).free
        if available < target_audio_bytes + MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "Gate D disk gate failed: "
                f"available={available}, candidate_upper_bound={target_audio_bytes}, "
                f"minimum_free={MINIMUM_FREE_BYTES}"
            )
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_build",
            [
                str(PYTHON),
                "scripts/build_model_ready_gate_d_tar.py",
                "--model-ready-dir",
                str(model_ready),
                "--target-audio-dir",
                str(target_audio),
                "--selection-list",
                str(gate_d_selection),
                "--output-tar",
                str(gate_d_tar),
                "--manifest",
                str(gate_d_build_manifest),
            ],
            report_artifact("gate_d_build.log"),
        )
        gate_d_build = load_json(gate_d_build_manifest)
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_score",
            [
                str(PYTHON),
                "scripts/run_audio_leakage_gate_v2_2.py",
                "score-tar",
                "--candidate-tar",
                str(gate_d_tar),
                "--benchmark-records",
                str(BENCHMARK_RECORDS),
                "--frozen-config",
                str(FROZEN_LEAKAGE_CONFIG),
                "--output-dir",
                str(gate_d_output),
            ],
            report_artifact("gate_d_scorer.log"),
        )
        gate_d_manifest = load_json(gate_d_output / "run_manifest.json")
        if not gate_d_manifest.get("gate_passed"):
            raise RuntimeError("final frozen Gate D did not pass")
        closure_command = [
            str(PYTHON),
            "scripts/audit_model_ready_gate_d_closure.py",
            "--model-ready-dir",
            str(model_ready),
            "--model-ready-validation",
            str(model_validation),
            "--target-audio-manifest",
            str(target_audio / "audio_manifest.jsonl"),
            "--selection-list",
            str(gate_d_selection),
            "--candidate-tar",
            str(gate_d_tar),
            "--gate-output-dir",
            str(gate_d_output),
            "--benchmark-records",
            str(BENCHMARK_RECORDS),
            "--frozen-config",
            str(FROZEN_LEAKAGE_CONFIG),
            "--expected-view-count",
            str(gate_d_build["model_ready_view_count"]),
            "--expected-source-count",
            str(gate_d_build["source_id_count"]),
            "--excluded-source-id",
            excluded_source_id,
            "--output",
            str(gate_d_closure),
        ]
        run_logged(
            state,
            orchestration_manifest,
            "gate_d_closure_audit",
            closure_command,
            report_artifact("gate_d_closure_audit.log"),
        )
        for artifact in (
            state_labels,
            target_audio,
            paraformer,
            rendered,
            timelines,
            glm,
            model_ready,
        ):
            verify_checksums(artifact)
        closure = load_json(gate_d_closure)
        state["result"] = {
            "source_count": scan_summary["sources"]["total"],
            "target_view_count_before_downstream_quarantine": scan_summary["views"][
                "structurally_usable"
            ],
            "target_view_hours": scan_summary["views"]["total_duration_hours"],
            "missing_state_event_count": expected_missing,
            "model_ready_row_count": validation["row_count"],
            "model_ready_view_count": validation["source_view_count"],
            "model_ready_exported_chunk_count": validation["exported_chunk_count"],
            "model_ready_dir": str(model_ready),
            "gate_d_closure": str(gate_d_closure),
            "gate_d_closure_sha256": sha256_file(gate_d_closure),
            "gate_d_passed": closure.get("gate_passed") is True,
            "full_qwen_accepted_cost_usd": full_audit["accepted_response_usage"][
                "cost_usd"
            ],
        }
        update(
            state,
            orchestration_manifest,
            "complete",
            "final_frozen_gate_d_complete",
            completed_at_utc=utc_now(),
            current_log=None,
            current_command=None,
        )
        return state
    except BaseException as exc:
        update(
            state,
            orchestration_manifest,
            "failed",
            state.get("stage", "unknown"),
            failed_at_utc=utc_now(),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--run-variant")
    parser.add_argument("--excluded-source-id")
    parser.add_argument(
        "--qwen-route",
        choices=sorted(SUPPORTED_QWEN_ROUTES),
        default=QWEN_ROUTE,
        help="Explicit network route for ordinary Qwen stages; legacy default is unchanged.",
    )
    parser.add_argument("--resume-after-qwen-full-manifest", type=Path)
    parser.add_argument("--failed-pipeline-snapshot", type=Path)
    parser.add_argument(
        "--expected-result-route-count",
        action="append",
        default=[],
        metavar="ROUTE=COUNT",
    )
    args = parser.parse_args()
    if args.resume_after_qwen_full_manifest is not None:
        if args.qwen_route != QWEN_ROUTE:
            parser.error("--qwen-route is only valid for a new ordinary shard run")
        if None in (
            args.archive_path,
            args.run_variant,
            args.excluded_source_id,
            args.failed_pipeline_snapshot,
        ):
            parser.error(
                "resume requires --archive-path, --run-variant, --excluded-source-id, "
                "and --failed-pipeline-snapshot"
            )
        try:
            expected_route_counts = parse_route_count_arguments(
                args.expected_result_route_count
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not expected_route_counts:
            parser.error("resume requires at least one --expected-result-route-count")
        result = resume_pipeline_after_qwen_full(
            args.contract,
            args.resume_after_qwen_full_manifest,
            args.failed_pipeline_snapshot,
            archive_path=args.archive_path,
            run_variant=args.run_variant,
            excluded_source_id=args.excluded_source_id,
            expected_route_counts=expected_route_counts,
        )
    else:
        result = run_pipeline(
            args.contract,
            args.download_manifest,
            archive_path=args.archive_path,
            run_variant=args.run_variant,
            excluded_source_id=args.excluded_source_id,
            qwen_route=args.qwen_route,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
