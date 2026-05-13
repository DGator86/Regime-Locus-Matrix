#!/usr/bin/env python3
"""Rebuild ``data/processed/ledgers/*.csv`` from trade logs + challenge state."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm.notify.ledger_books import write_trading_ledgers  # noqa: E402


def main() -> int:
    for name, path in write_trading_ledgers(ROOT).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
