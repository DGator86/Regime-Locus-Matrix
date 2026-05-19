#!/usr/bin/env python3
"""Set ALPHA_VANTAGE_API_KEY in repo .env without printing the key.

Usage:
  python scripts/set_alpha_vantage_key.py YOUR_KEY
  # or
  ALPHA_VANTAGE_API_KEY=YOUR_KEY python scripts/set_alpha_vantage_key.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = REPO / ".env"
KEY = "ALPHA_VANTAGE_API_KEY"


def main() -> int:
    val = (os.environ.get(KEY) or (sys.argv[1] if len(sys.argv) > 1 else "")).strip()
    if not val:
        print(f"Usage: {KEY}=... python {Path(__file__).name}  OR  python {Path(__file__).name} <key>", file=sys.stderr)
        return 1
    lines: list[str] = []
    if ENV.is_file():
        lines = ENV.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{KEY}="):
            out.append(f"{KEY}={val}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{KEY}={val}")
    ENV.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"Updated {ENV} ({KEY} set, value not shown).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
