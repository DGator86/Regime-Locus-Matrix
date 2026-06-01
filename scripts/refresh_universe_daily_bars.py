#!/usr/bin/env python3
"""Refresh ``data/raw/bars_{SYMBOL}.csv`` for the liquid universe (yfinance + 1m lake)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlm.data.daily_bars_sync import refresh_universe_daily_bars  # noqa: E402
from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated tickers (default: EXPANDED_LIQUID_UNIVERSE)",
    )
    parser.add_argument("--data-root", default="", help="Override data root (default: repo/data)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else list(EXPANDED_LIQUID_UNIVERSE)
    )
    data_root = Path(args.data_root).expanduser() if args.data_root.strip() else ROOT / "data"

    results = refresh_universe_daily_bars(symbols, data_root)
    ok = 0
    for r in results:
        last = r.last_timestamp.date() if r.last_timestamp is not None else "N/A"
        tag = "1m-merge" if r.merged_from_1m else "yf"
        print(f"{r.symbol}: rows={r.total_rows} last={last} via={tag} path={r.bars_path}")
        if r.total_rows > 0:
            ok += 1

    print(f"[refresh-universe-bars] {ok}/{len(symbols)} symbols with daily bars")
    return 0 if ok == len(symbols) else 1


if __name__ == "__main__":
    raise SystemExit(main())
