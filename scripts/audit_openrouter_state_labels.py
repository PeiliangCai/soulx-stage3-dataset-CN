#!/usr/bin/env python3
"""Audit OpenRouter state-label results for mechanical and provenance closure."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.openrouter_client import (  # noqa: E402
    OpenRouterClient,
    SUPPORTED_NETWORK_ROUTE_POLICIES,
    validate_daily_budget_status,
)
from duplexconv_stage3.state_labeling import (  # noqa: E402
    load_api_key,
    read_jsonl,
    validate_structured_labels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_expected_route_counts(
    values: list[str], *, default_route: str, result_count: int
) -> Counter[str]:
    """Build an exact route-provenance contract without weakening legacy runs."""

    if not values:
        return Counter({default_route: result_count})
    expected: Counter[str] = Counter()
    for value in values:
        route, separator, raw_count = value.rpartition("=")
        if not separator or route not in SUPPORTED_NETWORK_ROUTE_POLICIES:
            raise ValueError(f"invalid expected route count: {value}")
        if route in expected:
            raise ValueError(f"duplicate expected route count: {route}")
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(f"invalid expected route count: {value}") from None
        if count < 0:
            raise ValueError(f"negative expected route count: {value}")
        expected[route] = count
    if sum(expected.values()) != result_count:
        raise ValueError(
            "expected route counts do not sum to the result count: "
            f"expected={sum(expected.values())}, results={result_count}"
        )
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-kind", choices=("calibration", "full"), required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-route", required=True)
    parser.add_argument(
        "--expected-route-count",
        action="append",
        default=[],
        metavar="ROUTE=COUNT",
        help=(
            "Require an exact mixed route-provenance distribution. When omitted, "
            "all results must use --expected-route, preserving legacy behavior."
        ),
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--calibration-answer-key", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    requests = list(read_jsonl(args.request_file))
    results = list(read_jsonl(args.result_file))
    run = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    request_by_signature = {record["request_signature"]: record for record in requests}

    request_signatures_unique = len(request_by_signature) == len(requests)
    result_signatures = [record["request_signature"] for record in results]
    signature_closure = (
        request_signatures_unique
        and len(result_signatures) == len(set(result_signatures))
        and set(result_signatures) == set(request_by_signature)
    )
    labels: list[dict] = []
    result_records_valid = True
    for result in results:
        request_record = request_by_signature.get(result.get("request_signature"))
        if request_record is None:
            result_records_valid = False
            continue
        try:
            validated = validate_structured_labels(
                {"labels": result["labels"]}, request_record["target_event_ids"]
            )
        except (KeyError, ValueError):
            result_records_valid = False
            continue
        if (
            result.get("kind") != args.expected_kind
            or result.get("source_id") != request_record["source_id"]
            or result.get("target_event_ids") != request_record["target_event_ids"]
        ):
            result_records_valid = False
        labels.extend(validated)

    requested_event_ids = [
        event_id for record in requests for event_id in record["target_event_ids"]
    ]
    accepted_event_ids = [label["event_id"] for label in labels]
    duplicate_event_id_count = len(accepted_event_ids) - len(set(accepted_event_ids))
    event_closure = (
        len(requested_event_ids) == len(set(requested_event_ids))
        and duplicate_event_id_count == 0
        and set(accepted_event_ids) == set(requested_event_ids)
    )
    model_counts = Counter(result.get("model") for result in results)
    route_counts = Counter(result.get("network_route_policy") for result in results)
    expected_route_counts = parse_expected_route_counts(
        args.expected_route_count,
        default_route=args.expected_route,
        result_count=len(results),
    )
    provider_counts = Counter(result.get("provider") for result in results)
    state_counts = Counter(label["state"] for label in labels)
    schema_attempt_counts = Counter(str(result.get("schema_attempt")) for result in results)
    label_collapse = bool(labels) and len(state_counts) < 2

    usage = {
        "prompt_tokens": sum((result.get("usage") or {}).get("prompt_tokens") or 0 for result in results),
        "completion_tokens": sum((result.get("usage") or {}).get("completion_tokens") or 0 for result in results),
        "total_tokens": sum((result.get("usage") or {}).get("total_tokens") or 0 for result in results),
        "cost_usd": sum((result.get("usage") or {}).get("cost") or 0.0 for result in results),
    }
    budget = validate_daily_budget_status(
        OpenRouterClient(
            load_api_key(args.env_file), network_route_policy=args.expected_route
        ).key_status()
    )

    fixed_model = model_counts == Counter({args.expected_model: len(results)})
    fixed_route = route_counts == expected_route_counts
    mechanical_gate = all(
        (
            run.get("status") == "complete",
            len(requests) == len(results),
            signature_closure,
            result_records_valid,
            event_closure,
            fixed_model,
            fixed_route,
            not label_collapse,
        )
    )
    audit = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_mechanical_gate" if mechanical_gate else "failed_mechanical_gate",
        "mechanical_gate_passed": mechanical_gate,
        "request_count": len(requests),
        "result_count": len(results),
        "target_event_count": len(requested_event_ids),
        "accepted_label_count": len(labels),
        "duplicate_event_id_count": duplicate_event_id_count,
        "signature_closure_passed": signature_closure,
        "event_id_closure_passed": event_closure,
        "result_records_valid": result_records_valid,
        "fixed_model_passed": fixed_model,
        "fixed_route_passed": fixed_route,
        "label_collapse_detected": label_collapse,
        "label_collapse_rule": "fewer than two distinct output states",
        "model_counts": dict(sorted(model_counts.items())),
        "network_route_policy_counts": dict(sorted(route_counts.items())),
        "expected_network_route_policy_counts": dict(
            sorted(expected_route_counts.items())
        ),
        "provider_counts": dict(sorted(provider_counts.items())),
        "schema_attempt_counts": dict(sorted(schema_attempt_counts.items())),
        "state_distribution": dict(sorted(state_counts.items())),
        "accepted_response_usage": usage,
        "post_run_server_budget_status": {
            "usage_daily_usd": budget["usage_daily"],
            "limit_usd": budget["limit"],
            "limit_remaining_usd": budget["limit_remaining"],
            "limit_reset": budget["limit_reset"],
        },
        "request_file": str(args.request_file),
        "request_file_sha256": sha256(args.request_file),
        "result_file": str(args.result_file),
        "result_file_sha256": sha256(args.result_file),
        "run_manifest": str(args.run_manifest),
        "run_manifest_sha256": sha256(args.run_manifest),
    }

    if args.calibration_answer_key:
        answer_key = json.loads(args.calibration_answer_key.read_text(encoding="utf-8"))
        predicted = {label["event_id"]: label["state"] for label in labels}
        common = sorted(set(answer_key) & set(predicted))
        agreement = sum(answer_key[event_id] == predicted[event_id] for event_id in common)
        audit["hidden_official_label_diagnostic"] = {
            "role": "diagnostic_only_not_an_accuracy_gate",
            "reason": "DuplexConv official state labels are LLM-assisted rather than human gold; the gate checks mechanical validity and label collapse.",
            "agreement_count": agreement,
            "total_count": len(common),
            "agreement_rate": agreement / len(common) if common else None,
        }
        audit["status"] = (
            "passed_mechanical_gate_accuracy_diagnostic_only"
            if mechanical_gate
            else "failed_mechanical_gate"
        )
        audit["proceed_to_full_run"] = mechanical_gate
    else:
        audit["proceed_to_state_finalization"] = mechanical_gate

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if mechanical_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
