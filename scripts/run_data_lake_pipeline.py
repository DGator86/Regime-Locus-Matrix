#!/usr/bin/env python3
"""
Run the recommended fetch order into the canonical data lake (local Parquet only):

1. IBKR stock daily + intraday bars per symbol
2. Massive option contracts per underlying
3. (Optional) Massive option bars / quotes / trades for explicit tickers

Default symbols: SPY, QQQ, IWM, AAPL, TSLA, NVDA

Requires:
  - TWS/Gateway + pip install 'regime-locus-matrix[ibkr]' pyarrow
  - MASSIVE_API_KEY + pyarrow

Examples::

    python scripts/run_data_lake_pipeline.py --symbols SPY,QQQ
    python scripts/run_data_lake_pipeline.py --skip-stocks --contracts-only --symbols SPY
    python scripts/run_data_lake_pipeline.py --option-tickers O:SPY260619C00650000 \\
        --option-from 2026-01-01 --option-to 2026-04-01 --option-underlying SPY
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated equity underlyings for IBKR + Massive contract pulls",
    )
    p.add_argument("--skip-stocks", action="store_true")
    p.add_argument("--skip-contracts", action="store_true")
    p.add_argument("--contracts-only", action="store_true", help="Only step 2 (Massive contracts)")
    p.add_argument(
        "--stock-1d-duration", default="2 Y", help="IBKR duration for daily bars")
    p.add_argument("--stock-1d-slug", default="2y")
    p.add_argument("--stock-1m-duration", default="10 D")
    p.add_argument("--stock-1m-slug", default="10d")
    p.add_argument("--expiration-date", default=None, help="Filter Massive contracts (YYYY-MM-DD)")
    p.add_argument(
        "--option-tickers",
        default="",
        help="Comma-separated O: tickers for aggs (and optionally quotes/trades windows)",
    )
    p.add_argument("--option-underlying", default="SPY", help="data/options/{this}/… for option aggs")
    p.add_argument("--option-from", default="2026-01-01")
    p.add_argument("--option-to", default="2026-04-01")
    p.add_argument("--option-timespan", default="day", choices=("day", "minute", "hour"))
    p.add_argument("--skip-option-bars", action="store_true")
    p.add_argument("--fetch-quotes", action="store_true")
    p.add_argument("--fetch-trades", action="store_true")
    p.add_argument("--quote-window-gte", default="2026-03-20T13:30:00Z")
    p.add_argument("--quote-window-lt", default="2026-03-20T20:00:00Z")
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        print("No symbols", file=sys.stderr)
        return 1

    py = sys.executable

    if args.contracts_only:
        for s in syms:
            cmd = [
                py,
                "scripts/fetch_massive_contracts.py",
                "--underlying",
                s,
                "--limit",
                "1000",
            ]
            if args.expiration_date:
                cmd += ["--expiration-date", args.expiration_date]
            rc = _run(cmd)
            if rc != 0:
                return rc
        return 0

    if not args.skip_stocks:
        for s in syms:
            rc = _run(
                [
                    py,
                    "scripts/fetch_ibkr_stock_parquet.py",
                    s,
                    "--duration",
                    args.stock_1d_duration,
                    "--bar-size",
                    "1 day",
                    "--duration-slug",
                    args.stock_1d_slug,
                    "--interval",
                    "1d",
                ]
            )
            if rc != 0:
                return rc
            rc = _run(
                [
                    py,
                    "scripts/fetch_ibkr_stock_parquet.py",
                    s,
                    "--duration",
                    args.stock_1m_duration,
                    "--bar-size",
                    "1 min",
                    "--duration-slug",
                    args.stock_1m_slug,
                    "--interval",
                    "1m",
                ]
            )
            if rc != 0:
                return rc

    if not args.skip_contracts:
        for s in syms:
            cmd = [py, "scripts/fetch_massive_contracts.py", "--underlying", s, "--limit", "1000"]
            if args.expiration_date:
                cmd += ["--expiration-date", args.expiration_date]
            rc = _run(cmd)
            if rc != 0:
                return rc

    tickers = [t.strip() for t in args.option_tickers.split(",") if t.strip()]
    if tickers and not args.skip_option_bars:
        for ot in tickers:
            rc = _run(
                [
                    py,
                    "scripts/fetch_massive_option_bars.py",
                    "--option-ticker",
                    ot,
                    "--underlying-path",
                    args.option_underlying,
                    "--multiplier",
                    "1",
                    "--timespan",
                    args.option_timespan,
                    "--from",
                    args.option_from,
                    "--to",
                    args.option_to,
                ]
            )
            if rc != 0:
                return rc

    if tickers and args.fetch_quotes:
        for ot in tickers:
            rc = _run(
                [
                    py,
                    "scripts/fetch_massive_option_quotes.py",
                    "--option-ticker",
                    ot,
                    "--underlying-path",
                    args.option_underlying,
                    "--timestamp-gte",
                    args.quote_window_gte,
                    "--timestamp-lt",
                    args.quote_window_lt,
                ]
            )
            if rc != 0:
                return rc

    if tickers and args.fetch_trades:
        for ot in tickers:
            rc = _run(
                [
                    py,
                    "scripts/fetch_massive_option_trades.py",
                    "--option-ticker",
                    ot,
                    "--underlying-path",
                    args.option_underlying,
                    "--timestamp-gte",
                    args.quote_window_gte,
                    "--timestamp-lt",
                    args.quote_window_lt,
                ]
            )
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
