#!/usr/bin/env python3
"""Wait for the active uploader, then launch the independent remote verifier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload-pid", type=int, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected-progress-records", type=int, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("verifier_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.verifier_command and args.verifier_command[0] == "--":
        args.verifier_command = args.verifier_command[1:]
    if not args.verifier_command:
        parser.error("verifier command is required after --")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uploader_is_alive(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        value = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except FileNotFoundError:
        return False
    return "upload_duplexconv_baidu_release.py" in value


def progress_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_failure_once(path: Path, *, reason: str, progress_count: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    value = {
        "schema_version": 1,
        "status": "failed_closed",
        "failed_at_utc": utc_now(),
        "reason": reason,
        "progress_record_count": progress_count,
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
    }
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    last_count: int | None = None
    while True:
        count = progress_record_count(args.progress)
        if count != last_count:
            print(
                json.dumps(
                    {
                        "observed_at_utc": utc_now(),
                        "progress_record_count": count,
                        "completion_exists": args.completion.exists(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_count = count

        if args.completion.exists():
            if count != args.expected_progress_records:
                reason = (
                    "completion file exists but progress count differs: "
                    f"expected={args.expected_progress_records} actual={count}"
                )
                write_failure_once(args.failure_report, reason=reason, progress_count=count)
                raise RuntimeError(reason)
            break
        if not uploader_is_alive(args.upload_pid):
            reason = "uploader exited before creating completion file"
            write_failure_once(args.failure_report, reason=reason, progress_count=count)
            raise RuntimeError(reason)
        if time.monotonic() - started > args.timeout_seconds:
            reason = f"timed out after {args.timeout_seconds} seconds waiting for uploader"
            write_failure_once(args.failure_report, reason=reason, progress_count=count)
            raise TimeoutError(reason)
        time.sleep(args.poll_seconds)

    print("upload completion detected; starting independent verifier", flush=True)
    completed = subprocess.run(args.verifier_command, check=False)
    if completed.returncode != 0:
        reason = f"independent verifier exited with status {completed.returncode}"
        write_failure_once(
            args.failure_report,
            reason=reason,
            progress_count=progress_record_count(args.progress),
        )
        return completed.returncode
    print("independent verifier completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
