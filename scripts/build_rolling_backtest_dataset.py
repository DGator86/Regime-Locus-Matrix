"""Fetch historical equity bars (IBKR), build aligned option-chain rows, and a walk-forward manifest.

Example:

    python scripts/build_rolling_backtest_dataset.py --symbol SPY --start 2022-01-01 --fetch-ibkr
    python scripts/build_rolling_backtest_dataset.py --symbol SPY --demo --end 2026-04-02

Outputs (default under repo ``data/``):

- ``data/raw/bars_{SYMBOL}.csv`` — OHLCV (+ vwap) indexed as ``timestamp`` column
- ``data/raw/option_chain_{SYMBOL}.csv`` — chain rows per bar date (synthetic grid; see module doc)
- ``data/processed/rolling_backtest_manifest.csv`` — IS/OOS index ranges per fold

Then:

    python scripts/run_walkforward.py --symbol SPY

For **real** option marks over time, append Massive snapshots (repeat daily with ``--as-of``):

    python scripts/append_option_snapshot.py --symbol SPY --as-of YYYY-MM-DD --replace-same-day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.backtest.walkforward import WalkForwardConfig
from rlm.datasets.backtest_data import (
    bars_to_csv_frame,
    fetch_ibkr_daily_bars_range,
    rolling_window_manifest,
    synthetic_bars_demo,
    synthetic_option_chain_from_bars,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY", help="Underlying ticker")
    p.add_argument("--start", type=str, default="2022-01-01", metavar="YYYY-MM-DD")
    p.add_argument("--end", type=str, default="", help="YYYY-MM-DD (default: today)")
    p.add_argument(
        "--fetch-ibkr",
        action="store_true",
        help="Download daily bars from IBKR (TWS/Gateway must be running).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use in-memory demo bars instead of IBKR (for CI / no TWS).",
    )
    p.add_argument("--warmup-days", type=int, default=800, help="Demo mode: number of daily bars")
    p.add_argument(
        "--chunk-days", type=int, default=365, help="IBKR chunk size per historical request"
    )
    p.add_argument("--is-window", type=int, default=100)
    p.add_argument("--oos-window", type=int, default=50)
    p.add_argument("--step-size", type=int, default=50)
    p.add_argument("--out-raw", type=str, default="data/raw")
    p.add_argument("--out-processed", type=str, default="data/processed")
    p.add_argument(
        "--bars-in",
        type=str,
        default="",
        help="Existing bars CSV (timestamp + OHLCV). Skips fetch; still builds chain + manifest.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym = str(args.symbol).upper()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.today().normalize()

    if args.fetch_ibkr and args.demo:
        raise SystemExit("Use either --fetch-ibkr or --demo, not both.")
    if args.fetch_ibkr and args.bars_in:
        raise SystemExit("Use either --fetch-ibkr or --bars-in, not both.")

    raw_dir = ROOT / args.out_raw
    proc_dir = ROOT / args.out_processed
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    bars_path = raw_dir / f"bars_{sym}.csv"
    chain_path = raw_dir / f"option_chain_{sym}.csv"
    manifest_path = proc_dir / "rolling_backtest_manifest.csv"

    if args.bars_in:
        p = Path(args.bars_in)
        if not p.is_file():
            raise SystemExit(f"--bars-in not found: {p}")
        bars = pd.read_csv(p, parse_dates=["timestamp"])
        bars = bars.sort_values("timestamp").set_index("timestamp")
    elif args.demo:
        span = max(1, (end - start).days + 5)
        periods = max(args.warmup_days, span, args.is_window + args.oos_window + 10)
        bars = synthetic_bars_demo(end, periods=periods)
        bars = bars[(bars.index >= start) & (bars.index <= end)]
    elif args.fetch_ibkr:
        print(
            f"Fetching IBKR daily {sym} {start.date()} .. {end.date()} (chunk={args.chunk_days}d)..."
        )
        bars = fetch_ibkr_daily_bars_range(
            sym,
            start=start,
            end=end,
            chunk_days=args.chunk_days,
        )
    else:
        raise SystemExit("Choose one: --fetch-ibkr, --demo, or --bars-in PATH")

    if bars.empty:
        raise SystemExit("No bars after build; check dates, IBKR session, or --bars-in content.")

    print(f"Bars: {len(bars)} rows from {bars.index.min().date()} to {bars.index.max().date()}")

    chain = synthetic_option_chain_from_bars(bars, underlying=sym)
    wf = WalkForwardConfig(
        is_window=args.is_window,
        oos_window=args.oos_window,
        step_size=args.step_size,
    )
    manifest = rolling_window_manifest(bars.index, wf)
    if manifest.empty:
        raise SystemExit(
            f"Need at least is_window+oos_window={args.is_window + args.oos_window} bars; got {len(bars)}."
        )

    bars_to_csv_frame(bars).to_csv(bars_path, index=False)
    chain.to_csv(chain_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote {bars_path.relative_to(ROOT)}")
    print(f"Wrote {chain_path.relative_to(ROOT)} ({len(chain)} chain rows)")
    print(f"Wrote {manifest_path.relative_to(ROOT)} ({len(manifest)} walk-forward windows)")
    print(
        "Next: python scripts/run_walkforward.py --bars {0} --chain {1}".format(
            bars_path, chain_path
        )
    )


if __name__ == "__main__":
    main()
