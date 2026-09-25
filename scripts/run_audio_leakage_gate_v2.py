#!/usr/bin/env python3
"""Build, calibrate, or apply the low-information-aware leakage gate v2."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.audio_leakage_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
