#!/usr/bin/env python3
"""Set EODHD_API_KEY in repo .env (gitignored)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
_KEY_LINE = re.compile(r"^EODHD_API_KEY=.*$", re.MULTILINE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("api_key", nargs="?", help="EODHD API token (or pass via EODHD_API_KEY env)")
    args = ap.parse_args()
    import os

    key = (args.api_key or os.environ.get("EODHD_API_KEY") or "").strip()
    if not key:
        print("Provide api_key argument or EODHD_API_KEY env", file=sys.stderr)
        return 1

    text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.is_file() else ""
    line = f"EODHD_API_KEY={key}\n"
    if _KEY_LINE.search(text):
        text = _KEY_LINE.sub(f"EODHD_API_KEY={key}", text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += line
    ENV_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote EODHD_API_KEY to {ENV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
