#!/usr/bin/env python3
"""Refresh ``data/health_snapshot/bars_SPY_slice.csv`` from ``data/raw/bars_SPY.csv``.

Run from repo root after updating raw bars. Preserves ``manifest.json`` row count.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bars_SPY.csv"
SNAP = ROOT / "data" / "health_snapshot"
OUT = SNAP / "bars_SPY_slice.csv"
MAN = SNAP / "manifest.json"
N_DATA = 65


def main() -> int:
    if not RAW.is_file():
        print(f"Missing {RAW}", file=sys.stderr)
        return 2
    lines = RAW.read_text(encoding="utf-8").splitlines()
    if len(lines) < N_DATA + 1:
        print("bars_SPY.csv too short", file=sys.stderr)
        return 2
    header = lines[0]
    body = lines[-(N_DATA) :]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
    if MAN.is_file():
        man = json.loads(MAN.read_text(encoding="utf-8"))
        man["bars_rows"] = N_DATA
        MAN.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({N_DATA} data rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
