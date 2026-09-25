"""Build and audit immutable, append-friendly Stage 3 aggregate views."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence


SHARED_CONTRACT_KEYS = (
    "schema_version",
    "columns",
    "prefix",
    "sequence_profile",
    "chunk_group",
    "max_token_length",
    "timeline_profile",
    "glm_audio_profile",
    "audio_token_raw_range",
    "token_ids",
    "upstream_commit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            result.append(value)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def verify_checksum_manifest(root: Path) -> None:
    manifest = root / "checksums.sha256"
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {manifest}:{line_number}") from exc
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch: {path}")


def _zero_audio_hits(closure: dict[str, Any]) -> bool:
    comparison = closure.get("audio_comparison", {})
    return all(
        comparison.get(key) == 0
        for key in (
            "exact_audio_match_count",
            "normalized_audio_match_count",
            "content_exact_match_count",
            "window_match_count",
            "near_duplicate_gate_hit_count",
        )
    )


def _single_parquet(model_ready_dir: Path) -> Path:
    paths = sorted((model_ready_dir / "data").glob("*.parquet"))
    if len(paths) != 1:
        raise ValueError(
            f"an input shard must contain exactly one Parquet file: {model_ready_dir}"
        )
    return paths[0]


def verify_shard(spec: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    shard_id = spec.get("shard_id")
    if not isinstance(shard_id, str) or not shard_id:
        raise ValueError("shard_id is required")
    model_ready_dir = Path(spec["model_ready_dir"]).resolve(strict=True)
    validation_path = Path(spec["validation_report"]).resolve(strict=True)
    closure_path = Path(spec["gate_d_closure"]).resolve(strict=True)
    verify_checksum_manifest(model_ready_dir)

    contract_path = model_ready_dir / "contract.json"
    stats_path = model_ready_dir / "stats.json"
    windows_path = model_ready_dir / "metadata" / "windows.jsonl"
    chunks_path = model_ready_dir / "quarantine" / "chunks.jsonl"
    views_path = model_ready_dir / "quarantine" / "views.jsonl"
    parquet_path = _single_parquet(model_ready_dir)
    contract = read_json(contract_path)
    stats = read_json(stats_path)
    validation = read_json(validation_path)
    closure = read_json(closure_path)

    if validation.get("status") != "passed":
        raise ValueError(f"loader validation did not pass: {shard_id}")
    if validation.get("checksum_failure_count") != 0:
        raise ValueError(f"loader checksum failures are nonzero: {shard_id}")
    if validation.get("global_view_and_chunk_closure_passed") is not True:
        raise ValueError(f"loader partition did not close: {shard_id}")
    if validation.get("export_stats_sha256") != sha256_file(stats_path):
        raise ValueError(f"loader validation is stale: {shard_id}")
    if closure.get("gate_passed") is not True or not _zero_audio_hits(closure):
        raise ValueError(f"final Gate D did not pass unchanged: {shard_id}")
    if closure.get("all_source_members_scored_exactly_twice") is not True:
        raise ValueError(f"Gate D scorer partition did not close: {shard_id}")
    if closure.get("selection_missing_count") != 0 or closure.get(
        "selection_extra_count"
    ) != 0:
        raise ValueError(f"Gate D selection did not close: {shard_id}")
    closure_hashes = closure.get("sha256", {})
    expected_hashes = {
        "model_ready_windows": sha256_file(windows_path),
        "model_ready_stats": sha256_file(stats_path),
        "model_ready_validation": sha256_file(validation_path),
    }
    for key, actual in expected_hashes.items():
        if closure_hashes.get(key) != actual:
            raise ValueError(f"Gate D {key} hash is stale: {shard_id}")

    table = pq.read_table(parquet_path, columns=["index", "sequence"])
    if table.column_names != ["index", "sequence"]:
        raise ValueError(f"unexpected Parquet columns: {shard_id}")
    indexes = table.column("index").to_pylist()
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"duplicate indexes inside shard: {shard_id}")
    windows = read_jsonl(windows_path)
    window_indexes = [item.get("index") for item in windows]
    if len(window_indexes) != len(set(window_indexes)) or set(window_indexes) != set(
        indexes
    ):
        raise ValueError(f"Parquet/metadata index closure failed: {shard_id}")
    if len(indexes) != stats.get("row_count"):
        raise ValueError(f"row count differs from stats: {shard_id}")
    view_ids = {item["view_id"] for item in windows}
    source_ids = {item["source_id"] for item in windows}
    if len(view_ids) != stats.get("source_view_count"):
        raise ValueError(f"view count differs from stats: {shard_id}")
    if closure.get("model_ready_view_count") != len(view_ids) or closure.get(
        "model_ready_source_id_count"
    ) != len(source_ids):
        raise ValueError(f"Gate D view/source counts are stale: {shard_id}")

    chunk_quarantine = read_jsonl(chunks_path)
    local_quarantined_chunks = sum(
        item["chunk_range"][1] - item["chunk_range"][0]
        for item in chunk_quarantine
    )
    if local_quarantined_chunks != stats.get("quarantined_chunk_count"):
        raise ValueError(f"local chunk quarantine differs from stats: {shard_id}")
    view_quarantine = read_jsonl(views_path)
    quarantined_view_ids = [item["view_id"] for item in view_quarantine]
    if len(quarantined_view_ids) != len(set(quarantined_view_ids)):
        raise ValueError(f"duplicate source-view quarantine: {shard_id}")
    if view_ids & set(quarantined_view_ids):
        raise ValueError(f"exported/quarantined views overlap: {shard_id}")
    source_view_quarantined_chunks = sum(
        item["original_chunk_count"] for item in view_quarantine
    )
    source_view_quarantined_events = sum(item["event_count"] for item in view_quarantine)

    return {
        "shard_id": shard_id,
        "model_ready_dir": str(model_ready_dir),
        "validation_report": str(validation_path),
        "gate_d_closure": str(closure_path),
        "contract": contract,
        "stats": stats,
        "indexes": indexes,
        "view_ids": sorted(view_ids),
        "source_ids": sorted(source_ids),
        "quarantined_view_ids": sorted(quarantined_view_ids),
        "source_view_quarantined_chunks": source_view_quarantined_chunks,
        "source_view_quarantined_events": source_view_quarantined_events,
        "parquet_path": str(parquet_path),
        "windows_path": str(windows_path),
        "chunks_path": str(chunks_path),
        "views_path": str(views_path) if views_path.exists() else None,
        "sha256": {
            "contract": sha256_file(contract_path),
            "stats": sha256_file(stats_path),
            "parquet": sha256_file(parquet_path),
            "windows": sha256_file(windows_path),
            "chunks": sha256_file(chunks_path),
            "views": sha256_file(views_path) if views_path.exists() else None,
            "validation": sha256_file(validation_path),
            "gate_d_closure": sha256_file(closure_path),
        },
    }


def _assert_disjoint(shards: Sequence[dict[str, Any]], field: str) -> None:
    seen: set[str] = set()
    for shard in shards:
        current = set(shard[field])
        overlap = seen & current
        if overlap:
            raise ValueError(
                f"cross-shard duplicate {field}: {sorted(overlap)[:10]}"
            )
        seen.update(current)


def _shared_contract(shards: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base = shards[0]["contract"]
    for shard in shards[1:]:
        current = shard["contract"]
        mismatches = [
            key for key in SHARED_CONTRACT_KEYS if current.get(key) != base.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"cross-shard contract mismatch for {shard['shard_id']}: {mismatches}"
            )
    return {key: base.get(key) for key in SHARED_CONTRACT_KEYS}


def _link(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    (root / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in paths),
        encoding="utf-8",
    )


def aggregate(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = read_json(config_path)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output_dir}")
    specs = config.get("shards")
    if not isinstance(specs, list) or not specs:
        raise ValueError("aggregate config requires a non-empty shards list")
    ids = [item.get("shard_id") for item in specs]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("shard IDs must be unique and lexicographically sorted")
    shards = [verify_shard(item) for item in specs]
    for field in ("indexes", "view_ids", "source_ids", "quarantined_view_ids"):
        _assert_disjoint(shards, field)
    exported_views = {item for shard in shards for item in shard["view_ids"]}
    quarantined_views = {
        item for shard in shards for item in shard["quarantined_view_ids"]
    }
    if exported_views & quarantined_views:
        raise ValueError("cross-shard exported/quarantined views overlap")
    shared_contract = _shared_contract(shards)

    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    (temporary / "data").mkdir(parents=True)
    (temporary / "metadata" / "by_shard").mkdir(parents=True)
    (temporary / "quarantine" / "by_shard").mkdir(parents=True)
    try:
        registry = []
        rows_by_ntrack: Counter[str] = Counter()
        state_counts: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        max_token_length = 0
        for shard in shards:
            shard_id = shard["shard_id"]
            parquet_source = Path(shard["parquet_path"])
            parquet_name = f"train-{shard_id.lower()}-{shard['sha256']['parquet'][:12]}.parquet"
            windows_name = f"{shard_id}.windows.jsonl"
            chunks_name = f"{shard_id}.chunks.jsonl"
            views_name = f"{shard_id}.views.jsonl"
            _link(parquet_source, temporary / "data" / parquet_name)
            _link(Path(shard["windows_path"]), temporary / "metadata" / "by_shard" / windows_name)
            _link(Path(shard["chunks_path"]), temporary / "quarantine" / "by_shard" / chunks_name)
            if shard["views_path"]:
                _link(
                    Path(shard["views_path"]),
                    temporary / "quarantine" / "by_shard" / views_name,
                )
            else:
                (temporary / "quarantine" / "by_shard" / views_name).touch()

            stats = shard["stats"]
            local_quarantined = stats["quarantined_chunk_count"]
            source_view_quarantined = len(shard["quarantined_view_ids"])
            source_view_quarantined_chunks = shard["source_view_quarantined_chunks"]
            source_view_quarantined_events = shard["source_view_quarantined_events"]
            for key in ("row_count", "source_view_count", "exported_chunk_count"):
                totals[key] += stats[key]
            totals["quarantined_chunk_count"] += local_quarantined
            totals["source_view_quarantined_count"] += source_view_quarantined
            totals["source_view_quarantined_chunk_count"] += source_view_quarantined_chunks
            totals["source_view_quarantined_event_count"] += source_view_quarantined_events
            rows_by_ntrack.update(
                {str(key): value for key, value in stats.get("rows_by_ntrack", {}).items()}
            )
            state_counts.update(stats.get("chunk_state_counts_before_window_quarantine", {}))
            max_token_length = max(
                max_token_length, stats.get("max_observed_tokenized_length", 0)
            )
            registry.append(
                {
                    "shard_id": shard_id,
                    "model_ready_dir": shard["model_ready_dir"],
                    "dataset_version": shard["contract"].get("dataset_version"),
                    "index_prefix": shard["contract"].get("index_prefix"),
                    "row_count": stats["row_count"],
                    "source_conversation_count": len(shard["source_ids"]),
                    "source_view_count": stats["source_view_count"],
                    "source_view_quarantined_count": source_view_quarantined,
                    "exported_chunk_count": stats["exported_chunk_count"],
                    "local_quarantined_chunk_count": local_quarantined,
                    "source_view_quarantined_chunk_count": source_view_quarantined_chunks,
                    "data_file": f"data/{parquet_name}",
                    "metadata_file": f"metadata/by_shard/{windows_name}",
                    "chunk_quarantine_file": f"quarantine/by_shard/{chunks_name}",
                    "view_quarantine_file": f"quarantine/by_shard/{views_name}",
                    "validation_report": shard["validation_report"],
                    "gate_d_closure": shard["gate_d_closure"],
                    "sha256": shard["sha256"],
                }
            )

        stats = {
            "schema_version": 1,
            "dataset_version": config["dataset_version"],
            "aggregate_profile": "immutable-multi-parquet-manifest-v1",
            "aggregate_shard_count": len(shards),
            "row_count": totals["row_count"],
            "source_conversation_count": len(
                {item for shard in shards for item in shard["source_ids"]}
            ),
            "input_source_view_count": totals["source_view_count"]
            + totals["source_view_quarantined_count"],
            "source_view_count": totals["source_view_count"],
            "source_view_quarantined_count": totals["source_view_quarantined_count"],
            "source_view_quarantined_chunk_count": totals[
                "source_view_quarantined_chunk_count"
            ],
            "source_view_quarantined_event_count": totals[
                "source_view_quarantined_event_count"
            ],
            "input_total_chunk_count": totals["exported_chunk_count"]
            + totals["quarantined_chunk_count"]
            + totals["source_view_quarantined_chunk_count"],
            "total_effective_chunk_count": totals["exported_chunk_count"]
            + totals["quarantined_chunk_count"],
            "exported_chunk_count": totals["exported_chunk_count"],
            "quarantined_chunk_count": totals["quarantined_chunk_count"],
            "total_quarantined_chunk_count": totals["quarantined_chunk_count"]
            + totals["source_view_quarantined_chunk_count"],
            "max_observed_tokenized_length": max_token_length,
            "rows_by_ntrack": dict(sorted(rows_by_ntrack.items())),
            "chunk_state_counts_before_window_quarantine": dict(
                sorted(state_counts.items())
            ),
        }
        contract = {
            **shared_contract,
            "dataset_version": config["dataset_version"],
            "aggregate_profile": "immutable-multi-parquet-manifest-v1",
            "append_policy": "create a new immutable version manifest; never mutate an existing aggregate",
            "training_split_status": "not_created",
            "input_shard_count": len(shards),
        }
        write_jsonl(temporary / "shard_registry.jsonl", registry)
        write_json(temporary / "stats.json", stats)
        write_json(temporary / "contract.json", contract)
        manifest = {
            "schema_version": 1,
            "profile": "duplexconv-stage3-aggregate-manifest-v1",
            "created_at_utc": utc_now(),
            "dataset_version": config["dataset_version"],
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "shard_count": len(shards),
            "shard_ids": ids,
            "shard_registry_identity_sha256": canonical_sha256(registry),
            "aggregate_stats": stats,
            "cross_shard_index_overlap_count": 0,
            "cross_shard_view_overlap_count": 0,
            "cross_shard_source_overlap_count": 0,
            "exported_quarantined_view_overlap_count": 0,
            "all_input_loader_validations_passed": True,
            "all_input_gate_d_closures_passed": True,
            "all_input_gate_d_audio_hit_counts_zero": True,
            "gate_d_transitive_union_closed": True,
            "training_split_created": False,
            "training_authorized": False,
        }
        write_json(temporary / "aggregate_manifest.json", manifest)
        _write_checksums(temporary)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_aggregate(
    *, config_path: Path, aggregate_dir: Path, report_path: Path
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    config = read_json(config_path.resolve(strict=True))
    aggregate_dir = aggregate_dir.resolve(strict=True)
    report_path = report_path.absolute()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {report_path}")
    verify_checksum_manifest(aggregate_dir)
    registry = read_jsonl(aggregate_dir / "shard_registry.jsonl")
    stats = read_json(aggregate_dir / "stats.json")
    manifest = read_json(aggregate_dir / "aggregate_manifest.json")
    expected_ids = [item["shard_id"] for item in config["shards"]]
    actual_ids = [item["shard_id"] for item in registry]
    if expected_ids != actual_ids or manifest.get("shard_ids") != expected_ids:
        raise ValueError("aggregate shard registry differs from frozen config")

    indexes: list[str] = []
    metadata_indexes: list[str] = []
    view_ids: set[str] = set()
    source_ids: set[str] = set()
    quarantined_view_ids: set[str] = set()
    local_quarantined_chunks = 0
    source_view_quarantined_chunks = 0
    source_view_quarantined_events = 0
    hardlink_checks = []
    for record, spec in zip(registry, config["shards"]):
        data_file = aggregate_dir / record["data_file"]
        metadata_file = aggregate_dir / record["metadata_file"]
        chunks_file = aggregate_dir / record["chunk_quarantine_file"]
        views_file = aggregate_dir / record["view_quarantine_file"]
        source_parquet = _single_parquet(Path(spec["model_ready_dir"]))
        hardlink_checks.append(
            data_file.stat().st_dev == source_parquet.stat().st_dev
            and data_file.stat().st_ino == source_parquet.stat().st_ino
        )
        if sha256_file(data_file) != record["sha256"]["parquet"]:
            raise ValueError(f"aggregate Parquet hash mismatch: {record['shard_id']}")
        table = pq.read_table(data_file, columns=["index"])
        indexes.extend(table.column("index").to_pylist())
        windows = read_jsonl(metadata_file)
        metadata_indexes.extend(item["index"] for item in windows)
        for item in windows:
            view_ids.add(item["view_id"])
            source_ids.add(item["source_id"])
        local_quarantined_chunks += sum(
            item["chunk_range"][1] - item["chunk_range"][0]
            for item in read_jsonl(chunks_file)
        )
        for item in read_jsonl(views_file):
            quarantined_view_ids.add(item["view_id"])
            source_view_quarantined_chunks += item["original_chunk_count"]
            source_view_quarantined_events += item["event_count"]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(metadata_indexes):
        raise ValueError("aggregate index partition did not close")
    if view_ids & quarantined_view_ids:
        raise ValueError("aggregate exported/quarantined views overlap")
    expected_stats = {
        "row_count": len(indexes),
        "source_conversation_count": len(source_ids),
        "source_view_count": len(view_ids),
        "source_view_quarantined_count": len(quarantined_view_ids),
        "source_view_quarantined_chunk_count": source_view_quarantined_chunks,
        "source_view_quarantined_event_count": source_view_quarantined_events,
        "quarantined_chunk_count": local_quarantined_chunks,
    }
    mismatches = {
        key: (stats.get(key), value)
        for key, value in expected_stats.items()
        if stats.get(key) != value
    }
    if mismatches:
        raise ValueError(f"aggregate stats mismatch: {mismatches}")
    report = {
        "schema_version": 1,
        "profile": "duplexconv-stage3-aggregate-independent-audit-v1",
        "completed_at_utc": utc_now(),
        "status": "passed",
        "aggregate_dir": str(aggregate_dir),
        "shard_count": len(registry),
        **expected_stats,
        "index_count": len(indexes),
        "unique_index_count": len(set(indexes)),
        "metadata_index_count": len(metadata_indexes),
        "all_local_data_files_are_hardlinks": all(hardlink_checks),
        "all_checksum_manifest_entries_passed": True,
        "all_input_gate_d_closures_passed": manifest.get(
            "all_input_gate_d_closures_passed"
        )
        is True,
        "gate_d_transitive_union_closed": manifest.get(
            "gate_d_transitive_union_closed"
        )
        is True,
        "aggregate_manifest_sha256": sha256_file(
            aggregate_dir / "aggregate_manifest.json"
        ),
        "aggregate_stats_sha256": sha256_file(aggregate_dir / "stats.json"),
        "checksums_sha256": sha256_file(aggregate_dir / "checksums.sha256"),
    }
    if not all(
        (
            report["all_local_data_files_are_hardlinks"],
            report["all_input_gate_d_closures_passed"],
            report["gate_d_transitive_union_closed"],
        )
    ):
        raise ValueError("aggregate audit did not close")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    return report
