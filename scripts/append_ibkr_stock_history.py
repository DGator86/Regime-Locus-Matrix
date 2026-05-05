#!/usr/bin/env python3
"""Append/merge IBKR stock bars into local parquet history without truncation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.lake import stock_1d_parquet


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    ex = existing[cols].copy() if not existing.empty else pd.DataFrame(columns=cols)
    fr = fresh[cols].copy() if not fresh.empty else pd.DataFrame(columns=cols)
    df = pd.concat([ex, fr], ignore_index=True)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--duration", default="30 D", help="IBKR duration fetch window")
    ap.add_argument("--bar-size", default="1 day")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        out = stock_1d_parquet(sym, duration_slug="full", root=ROOT)
        fresh = fetch_historical_stock_bars(sym, duration=args.duration, bar_size=args.bar_size, timeout_sec=120.0)
        existing = pd.read_parquet(out) if out.is_file() else pd.DataFrame()
        merged = _merge(existing, fresh)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, index=False)
        print(f"[append-stock] {sym}: existing={len(existing)} fresh={len(fresh)} merged={len(merged)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
