#!/usr/bin/env python3
"""Read-only bootstrap summary for long-running project tasks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "project_state/active_tasks.json"


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def tmux_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def main() -> int:
    registry = load_json(REGISTRY)
    if not registry:
        raise FileNotFoundError(REGISTRY)
    print(registry["role"])
    for task in registry["tasks"]:
        print(f"\n[{task['task_id']}]")
        source = task.get("status_source")
        if source:
            path = Path(source)
            live = load_json(path)
            print(f"manifest={path}")
            if live is None:
                print("live_status=manifest_missing")
            else:
                print(f"live_status={live.get('status', 'unknown')}")
                completed = live.get("completed_classes")
                if isinstance(completed, list):
                    print(f"completed_classes={len(completed)}")
                for key in ("updated_at_utc", "completed_at_utc"):
                    if key in live:
                        print(f"{key}={live.get(key)}")
        else:
            print(f"registry_status={task.get('status', 'unknown')}")
        tmux = task.get("tmux_session")
        if tmux:
            print(f"tmux={tmux} exists={str(tmux_exists(tmux)).lower()}")
        print(f"next_action={task.get('next_action', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
