#!/usr/bin/env python3
"""Execute frozen OpenRouter requests with strict cache validation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.openrouter_client import main


if __name__ == "__main__":
    raise SystemExit(main())
