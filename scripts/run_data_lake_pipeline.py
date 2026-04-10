#!/usr/bin/env python3
"""Run the recommended fetch order into the canonical data lake (local Parquet only)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.factors.multi_timeframe import format_precompute_instructions, parse_higher_tfs
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.utils.parallel import parallel_map

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_bundle(cmds: list[list[str]]) -> int:
    for cmd in cmds:
        rc = _run(cmd)
        if rc != 0:
            return rc
    return 0


def _stock_task(task: tuple[str, str, str, str, str, str]) -> int:
    py, symbol, stock_1d_duration, stock_1d_slug, stock_1m_duration, stock_1m_slug = task
    return _run_bundle(
        [
            [
                py,
                "scripts/fetch_ibkr_stock_parquet.py",
                symbol,
                "--duration",
                stock_1d_duration,
                "--bar-size",
                "1 day",
                "--duration-slug",
                stock_1d_slug,
                "--interval",
                "1d",
            ],
            [
                py,
                "scripts/fetch_ibkr_stock_parquet.py",
                symbol,
                "--duration",
                stock_1m_duration,
                "--bar-size",
                "1 min",
                "--duration-slug",
                stock_1m_slug,
                "--interval",
                "1m",
            ],
        ]
    )


def _contract_task(task: tuple[str, str, str | None]) -> int:
    py, symbol, expiration_date = task
    cmd = [py, "scripts/fetch_massive_contracts.py", "--underlying", symbol, "--limit", "1000"]
    if expiration_date:
        cmd += ["--expiration-date", expiration_date]
    return _run(cmd)


def _option_bars_task(task: tuple[str, str, str, str, str, str]) -> int:
    py, ot, underlying, timespan, opt_from, opt_to = task
    return _run(
        [
            py,
            "scripts/fetch_massive_option_bars.py",
            "--option-ticker",
            ot,
            "--underlying-path",
            underlying,
            "--multiplier",
            "1",
            "--timespan",
            timespan,
            "--from",
            opt_from,
            "--to",
            opt_to,
        ]
    )


def _option_quotes_task(task: tuple[str, str, str, str, str]) -> int:
    py, ot, underlying, ts_gte, ts_lt = task
    return _run(
        [
            py,
            "scripts/fetch_massive_option_quotes.py",
            "--option-ticker",
            ot,
            "--underlying-path",
            underlying,
            "--timestamp-gte",
            ts_gte,
            "--timestamp-lt",
            ts_lt,
        ]
    )


def _option_trades_task(task: tuple[str, str, str, str, str]) -> int:
    py, ot, underlying, ts_gte, ts_lt = task
    return _run(
        [
            py,
            "scripts/fetch_massive_option_trades.py",
            "--option-ticker",
            ot,
            "--underlying-path",
            underlying,
            "--timestamp-gte",
            ts_gte,
            "--timestamp-lt",
            ts_lt,
        ]
    )


def _first_bad(results: list[int]) -> int:
    return next((int(rc) for rc in results if int(rc) != 0), 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated equity underlyings for IBKR + Massive contract pulls",
    )
    p.add_argument("--skip-stocks", action="store_true")
    p.add_argument("--skip-contracts", action="store_true")
    p.add_argument("--contracts-only", action="store_true", help="Only step 2 (Massive contracts)")
    p.add_argument("--stock-1d-duration", default="2 Y", help="IBKR duration for daily bars")
    p.add_argument("--stock-1d-slug", default="2y")
    p.add_argument("--stock-1m-duration", default="10 D")
    p.add_argument("--stock-1m-slug", default="10d")
    p.add_argument("--expiration-date", default=None, help="Filter Massive contracts (YYYY-MM-DD)")
    p.add_argument(
        "--option-tickers",
        default="",
        help="Comma-separated O: tickers for aggs (and optionally quotes/trades windows)",
    )
    p.add_argument(
        "--option-underlying", default="SPY", help="data/options/{this}/… for option aggs"
    )
    p.add_argument("--option-from", default="2026-01-01")
    p.add_argument("--option-to", default="2026-04-01")
    p.add_argument("--option-timespan", default="day", choices=("day", "minute", "hour"))
    p.add_argument("--skip-option-bars", action="store_true")
    p.add_argument("--fetch-quotes", action="store_true")
    p.add_argument("--fetch-trades", action="store_true")
    p.add_argument("--quote-window-gte", default="2026-03-20T13:30:00Z")
    p.add_argument("--quote-window-lt", default="2026-03-20T20:00:00Z")
    p.add_argument("--mtf", action="store_true", help="Print and enforce MTF pre-compute guidance.")
    p.add_argument(
        "--higher-tfs",
        default="1W,1M",
        help="Comma-separated higher-timeframe resample rules for --mtf (example: 1W,1M).",
    )
    p.add_argument(
        "--jobs", type=int, default=1, help="Number of workers for per-symbol / per-ticker tasks"
    )
    p.add_argument(
        "--parallel-backend",
        default="process",
        choices=("serial", "thread", "process", "ray"),
        help="Execution backend for parallel ingest.",
    )
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        print("No symbols", file=sys.stderr)
        return 1
    higher_tfs = parse_higher_tfs(args.higher_tfs)
    if args.mtf:
        print("MTF mode enabled for downstream pipelines.")
        for sym in syms:
            print(format_precompute_instructions(symbol=sym, higher_tfs=higher_tfs))

    py = sys.executable

    if args.contracts_only:
        tasks = [(py, s, args.expiration_date) for s in syms]
        return _first_bad(
            parallel_map(
                _contract_task, tasks, max_workers=args.jobs, backend=args.parallel_backend
            )
        )

    if not args.skip_stocks:
        stock_tasks = [
            (
                py,
                s,
                args.stock_1d_duration,
                args.stock_1d_slug,
                args.stock_1m_duration,
                args.stock_1m_slug,
            )
            for s in syms
        ]
        bad = _first_bad(
            parallel_map(
                _stock_task, stock_tasks, max_workers=args.jobs, backend=args.parallel_backend
            )
        )
        if bad:
            return bad

    if not args.skip_contracts:
        contract_tasks = [(py, s, args.expiration_date) for s in syms]
        bad = _first_bad(
            parallel_map(
                _contract_task, contract_tasks, max_workers=args.jobs, backend=args.parallel_backend
            )
        )
        if bad:
            return bad

    tickers = [t.strip() for t in args.option_tickers.split(",") if t.strip()]
    if tickers and not args.skip_option_bars:
        bar_tasks = [
            (py, ot, args.option_underlying, args.option_timespan, args.option_from, args.option_to)
            for ot in tickers
        ]
        bad = _first_bad(
            parallel_map(
                _option_bars_task, bar_tasks, max_workers=args.jobs, backend=args.parallel_backend
            )
        )
        if bad:
            return bad

    if tickers and args.fetch_quotes:
        quote_tasks = [
            (py, ot, args.option_underlying, args.quote_window_gte, args.quote_window_lt)
            for ot in tickers
        ]
        bad = _first_bad(
            parallel_map(
                _option_quotes_task,
                quote_tasks,
                max_workers=args.jobs,
                backend=args.parallel_backend,
            )
        )
        if bad:
            return bad

    if tickers and args.fetch_trades:
        trade_tasks = [
            (py, ot, args.option_underlying, args.quote_window_gte, args.quote_window_lt)
            for ot in tickers
        ]
        bad = _first_bad(
            parallel_map(
                _option_trades_task,
                trade_tasks,
                max_workers=args.jobs,
                backend=args.parallel_backend,
            )
        )
        if bad:
            return bad

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
