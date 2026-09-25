#!/usr/bin/env python3
"""Stop the frozen A/B/C/D Table 3 run once only one checkpoint remains active."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


TARGET_SCRIPT_NAMES = (
    "run_abcd_table3_step5_10.py",
    "run_table3_reproduction.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tmux_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def matching_processes(output_root: Path) -> list[dict[str, Any]]:
    marker = str(output_root)
    records = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [
            part.decode("utf-8", errors="replace")
            for part in raw.split(b"\0")
            if part
        ]
        if not argv or "python" not in Path(argv[0]).name.lower():
            continue
        command = " ".join(argv)
        if marker not in command or not any(name in command for name in TARGET_SCRIPT_NAMES):
            continue
        records.append({"pid": int(entry.name), "command": command})
    return sorted(records, key=lambda row: row["pid"])


def stop_processes(records: list[dict[str, Any]], sig: signal.Signals) -> None:
    for record in records:
        try:
            os.kill(record["pid"], sig)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-tmux", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--required-consecutive", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    receipt_path = args.receipt.resolve()
    if args.poll_seconds <= 0 or args.required_consecutive <= 0:
        raise ValueError("poll interval and consecutive count must be positive")
    if receipt_path.exists():
        raise RuntimeError(f"receipt already exists: {receipt_path}")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "monitoring",
        "policy": "stop after exactly one active checkpoint is observed in consecutive manifest reads",
        "required_consecutive": args.required_consecutive,
        "poll_seconds": args.poll_seconds,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "target_tmux": args.target_tmux,
        "started_at_utc": utc_now(),
    }
    atomic_json_write(receipt_path, receipt)

    consecutive = 0
    while True:
        manifest = load_json(manifest_path)
        active = manifest.get("active_assignments") or []
        status = manifest.get("status")
        if status == "complete":
            receipt.update(
                {
                    "status": "run_completed_before_stop_condition",
                    "completed_at_utc": utc_now(),
                    "final_manifest_stage": manifest.get("stage"),
                }
            )
            atomic_json_write(receipt_path, receipt)
            return 0
        if not tmux_exists(args.target_tmux):
            receipt.update(
                {
                    "status": "target_tmux_disappeared_before_stop_condition",
                    "completed_at_utc": utc_now(),
                    "final_manifest_status": status,
                    "final_manifest_stage": manifest.get("stage"),
                }
            )
            atomic_json_write(receipt_path, receipt)
            return 2

        consecutive = consecutive + 1 if len(active) == 1 else 0
        receipt.update(
            {
                "last_observed_at_utc": utc_now(),
                "last_manifest_status": status,
                "last_manifest_stage": manifest.get("stage"),
                "last_active_assignments": active,
                "consecutive_single_active_observations": consecutive,
            }
        )
        atomic_json_write(receipt_path, receipt)
        if consecutive < args.required_consecutive:
            time.sleep(args.poll_seconds)
            continue

        receipt.update(
            {
                "status": "stopping",
                "stop_triggered_at_utc": utc_now(),
                "trigger_manifest_status": status,
                "trigger_manifest_stage": manifest.get("stage"),
                "trigger_active_assignments": active,
                "processes_before_stop": matching_processes(output_root),
            }
        )
        atomic_json_write(receipt_path, receipt)
        subprocess.run(["tmux", "kill-session", "-t", args.target_tmux], check=False)

        deadline = time.monotonic() + 20
        remaining = matching_processes(output_root)
        while remaining and time.monotonic() < deadline:
            time.sleep(1)
            remaining = matching_processes(output_root)
        if remaining:
            stop_processes(remaining, signal.SIGTERM)
            time.sleep(5)
            remaining = matching_processes(output_root)
        if remaining:
            stop_processes(remaining, signal.SIGKILL)
            time.sleep(1)
            remaining = matching_processes(output_root)

        receipt.update(
            {
                "status": "stopped" if not remaining else "stop_incomplete",
                "completed_at_utc": utc_now(),
                "processes_after_stop": remaining,
                "results_policy": "Complete class outputs remain reusable; any running class output must be archived by the existing explicit resume path.",
            }
        )
        atomic_json_write(receipt_path, receipt)
        return 0 if not remaining else 3


if __name__ == "__main__":
    raise SystemExit(main())
