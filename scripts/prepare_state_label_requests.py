#!/usr/bin/env python3
"""Prepare calibration, connectivity, and full OpenRouter requests."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.state_labeling import prepare_main


if __name__ == "__main__":
    raise SystemExit(prepare_main())
