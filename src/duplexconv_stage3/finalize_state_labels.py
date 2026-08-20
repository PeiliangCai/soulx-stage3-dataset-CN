"""Merge official, deterministic WAIT, and Qwen state supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from .source_scan import sha256_file
from .state_labeling import (
    ALLOWED_OUTPUT_STATES,
    MODEL_ID,
    MODEL_STANDARD_NAME,
    canonical_json,
    read_jsonl,
    validate_structured_labels,
)
from .openrouter_client import NETWORK_ROUTE_POLICY


def finalize_state_labels(
    *, scan_dir: Path, request_dir: Path, output_dir: Path
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve(strict=True)
    request_dir = request_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        events = list(read_jsonl(scan_dir / "events.jsonl"))
        requests = list(read_jsonl(request_dir / "full_requests.jsonl"))
        results = list(read_jsonl(request_dir / "full_results.jsonl"))
        request_by_signature = {
            item["request_signature"]: item for item in requests
        }
        if len(request_by_signature) != 404 or len(results) != 404:
            raise ValueError("request/result count is not 404")

        qwen_labels: dict[str, dict[str, Any]] = {}
        providers = Counter()
        schema_attempts = Counter()
        accepted_usage = Counter()
        openrouter_ids = set()
        for result in results:
            signature = result["request_signature"]
            if signature not in request_by_signature:
                raise ValueError(f"unknown request signature: {signature}")
            request_record = request_by_signature[signature]
            labels = validate_structured_labels(
                {"labels": result["labels"]}, request_record["target_event_ids"]
            )
            if result["model"] != MODEL_ID:
                raise ValueError(f"model drift: {result['model']} != {MODEL_ID}")
            if result.get("network_route_policy") != NETWORK_ROUTE_POLICY:
                raise ValueError("network route provenance is missing or wrong")
            providers[str(result.get("provider"))] += 1
            schema_attempts[str(result.get("schema_attempt", 0))] += 1
            if result.get("openrouter_id"):
                openrouter_ids.add(result["openrouter_id"])
            for key, value in (result.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    accepted_usage[key] += value
            for label in labels:
                event_id = label["event_id"]
                if event_id in qwen_labels:
                    raise ValueError(f"duplicate Qwen label: {event_id}")
                qwen_labels[event_id] = {
                    **label,
                    "source_id": result["source_id"],
                    "request_signature": signature,
                    "openrouter_id": result.get("openrouter_id"),
                    "model_standard_name": MODEL_STANDARD_NAME,
                    "model_api_slug": MODEL_ID,
                    "provider": result.get("provider"),
                    "network_route_policy": NETWORK_ROUTE_POLICY,
                    "prompt_version": request_record["prompt_version"],
                    "prompt_sha256": request_record["prompt_sha256"],
                    "schema_version": request_record["schema_version"],
                    "schema_sha256": request_record["schema_sha256"],
                }
        if len(qwen_labels) != 1599:
            raise ValueError(f"Qwen label count is {len(qwen_labels)}, expected 1599")

        finalized = []
        source_counts = Counter()
        final_distribution = Counter()
        for event in events:
            official = event["official_state"]
            event_id = event["event_id"]
            if official in ALLOWED_OUTPUT_STATES:
                final_state = official
                label_source = "duplexconv_official_llm_assisted"
                quality = "official_llm_assisted_not_human_gold"
                label_metadata = None
            elif official == "wait":
                final_state = "complete"
                label_source = "deterministic_wait_to_complete"
                quality = "deterministic_mapping"
                label_metadata = {
                    "original_state": "wait",
                    "mapped_state": "complete",
                    "wait_policy": "wait-to-complete-v1",
                }
            elif official is None:
                if event_id not in qwen_labels:
                    raise ValueError(f"missing Qwen label for {event_id}")
                final_state = qwen_labels[event_id]["state"]
                label_source = "openrouter_qwen3_235b_a22b_instruct_2507"
                quality = "qwen_pseudolabel_not_human_gold"
                label_metadata = qwen_labels[event_id]
            else:
                raise ValueError(f"unsupported official state for {event_id}: {official}")
            source_counts[label_source] += 1
            final_distribution[final_state] += 1
            finalized.append(
                {
                    **event,
                    "final_state": final_state,
                    "state_label_source": label_source,
                    "state_label_quality": quality,
                    "state_label_metadata": label_metadata,
                }
            )
        if len(finalized) != 8505:
            raise ValueError("final event count is not 8,505")
        if source_counts != Counter(
            {
                "duplexconv_official_llm_assisted": 6895,
                "deterministic_wait_to_complete": 11,
                "openrouter_qwen3_235b_a22b_instruct_2507": 1599,
            }
        ):
            raise ValueError(f"state source closure failed: {source_counts}")

        with (output_dir / "events_with_final_state.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for item in finalized:
                handle.write(canonical_json(item) + "\n")
        with (output_dir / "qwen_labels.jsonl").open("w", encoding="utf-8") as handle:
            for event_id in sorted(qwen_labels):
                handle.write(canonical_json(qwen_labels[event_id]) + "\n")
        summary = {
            "schema_version": 1,
            "status": "passed",
            "event_count": len(finalized),
            "state_source_counts": dict(sorted(source_counts.items())),
            "final_state_distribution": dict(sorted(final_distribution.items())),
            "qwen": {
                "model_standard_name": MODEL_STANDARD_NAME,
                "model_api_slug": MODEL_ID,
                "request_count": len(results),
                "event_count": len(qwen_labels),
                "network_route_policy": NETWORK_ROUTE_POLICY,
                "providers": dict(sorted(providers.items())),
                "schema_attempts": dict(sorted(schema_attempts.items())),
                "accepted_response_usage": dict(sorted(accepted_usage.items())),
                "accepted_openrouter_id_count": len(openrouter_ids),
                "daily_budget_usd": 10.0,
                "usage_note": "Accepted-response usage excludes any rejected response attempts; API-key usage_daily is the authoritative budget counter.",
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (output_dir / "checksums.sha256").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )
        return summary
    except Exception:
        # Formal output must never survive in a partial state.
        for path in sorted(output_dir.glob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        output_dir.rmdir()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize all 8,505 state events.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize_state_labels(
        scan_dir=args.scan_dir,
        request_dir=args.request_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
