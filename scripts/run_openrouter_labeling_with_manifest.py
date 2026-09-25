#!/usr/bin/env python3
"""Run frozen OpenRouter requests with an interruption-recovery manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.openrouter_client import (  # noqa: E402
    DAILY_BUDGET_CAP_USD,
    DEFAULT_NETWORK_ROUTE_POLICY,
    DIRECT_NETWORK_ROUTE_POLICY,
    _load_request_records,
    run_requests,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.manifest}")
    records = _load_request_records(args.request_file)
    if args.source_id:
        selected = set(args.source_id)
        records = [record for record in records if record["source_id"] in selected]
    if args.limit is not None:
        records = records[: args.limit]
    state: dict[str, Any] = {
        "schema_version": 1,
        "profile": "duplexconv-openrouter-labeling-run-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "request_file": str(args.request_file.resolve()),
        "request_file_sha256": sha256_file(args.request_file),
        "request_count": len(records),
        "target_event_count": sum(len(record["target_event_ids"]) for record in records),
        "request_kind_counts": dict(sorted(Counter(record["kind"] for record in records).items())),
        "models": sorted({record["model"] for record in records}),
        "prompt_versions": sorted({record["prompt_version"] for record in records}),
        "network_route_policy": args.network_route_policy,
        "workers": args.workers,
        "daily_budget_cap_usd": DAILY_BUDGET_CAP_USD,
        "server_side_budget_preflight_required": True,
        "full_run_confirmation_supplied": args.full_run_confirmation is not None,
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_file_count_before": (
            sum(1 for path in args.cache_dir.glob("*.json") if path.is_file())
            if args.cache_dir.exists()
            else 0
        ),
        "result_file": str(args.result_file.resolve()),
        "env_file": str(args.env_file.resolve()),
        "env_file_mode": oct(args.env_file.stat().st_mode & 0o777),
        "api_key_recorded_in_manifest": False,
    }
    atomic_json(args.manifest, state)
    try:
        summary = run_requests(
            args.request_file,
            args.cache_dir,
            args.result_file,
            args.env_file,
            limit=args.limit,
            full_run_confirmation=args.full_run_confirmation,
            workers=args.workers,
            source_ids=args.source_id,
            network_route_policy=args.network_route_policy,
        )
        state.update(
            {
                "status": "complete",
                "updated_at_utc": utc_now(),
                "completed_at_utc": utc_now(),
                "summary": summary,
                "result_file_sha256": sha256_file(args.result_file),
                "cache_file_count_after": sum(
                    1 for path in args.cache_dir.glob("*.json") if path.is_file()
                ),
            }
        )
        atomic_json(args.manifest, state)
        print(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2))
        return state
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "updated_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cache_file_count_after": (
                    sum(1 for path in args.cache_dir.glob("*.json") if path.is_file())
                    if args.cache_dir.exists()
                    else 0
                ),
            }
        )
        atomic_json(args.manifest, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full-run-confirmation")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-id", action="append")
    parser.add_argument(
        "--network-route-policy",
        choices=(DEFAULT_NETWORK_ROUTE_POLICY, DIRECT_NETWORK_ROUTE_POLICY),
        default=DEFAULT_NETWORK_ROUTE_POLICY,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
