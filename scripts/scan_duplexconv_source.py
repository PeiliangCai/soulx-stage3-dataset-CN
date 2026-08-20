#!/usr/bin/env python3
"""CLI entry point for the official DuplexConv source scan."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.source_scan import main


if __name__ == "__main__":
    raise SystemExit(main())
