#!/usr/bin/env python3
"""Run ingestion pipeline into the canonical local data lake."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.ingestion.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
