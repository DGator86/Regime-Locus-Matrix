"""Ingestion pipeline orchestrator for data lake population."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from rlm.features.factors.multi_timeframe import format_precompute_instructions, parse_higher_tfs
from rlm.ingestion.config import IngestionConfig
from rlm.ingestion.writers import (
    write_ibkr_stock_parquet,
    write_massive_option_bars_parquet,
    write_massive_option_contracts_parquet,
    write_massive_option_quotes_parquet,
    write_massive_option_trades_parquet,
)
from rlm.utils.parallel import parallel_map

DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA")


@dataclass
class IngestionPipeline:
    config: IngestionConfig = field(default_factory=IngestionConfig)

    def run(self, args: argparse.Namespace) -> int:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not syms:
            return 1

        higher_tfs = parse_higher_tfs(args.higher_tfs)
        if args.mtf:
            print("MTF mode enabled for downstream pipelines.")
            for sym in syms:
                print(format_precompute_instructions(symbol=sym, higher_tfs=higher_tfs))

        if args.contracts_only:
            return _first_bad(
                parallel_map(
                    _contract_task,
                    [(s, args.expiration_date) for s in syms],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )

        if not args.skip_stocks:
            bad = _first_bad(
                parallel_map(
                    _stock_task,
                    [
                        (
                            s,
                            args.stock_1d_duration,
                            args.stock_1d_slug,
                            args.stock_1m_duration,
                            args.stock_1m_slug,
                        )
                        for s in syms
                    ],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )
            if bad:
                return bad

        if not args.skip_contracts:
            bad = _first_bad(
                parallel_map(
                    _contract_task,
                    [(s, args.expiration_date) for s in syms],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )
            if bad:
                return bad

        tickers = [t.strip() for t in args.option_tickers.split(",") if t.strip()]
        if tickers and not args.skip_option_bars:
            bad = _first_bad(
                parallel_map(
                    _option_bars_task,
                    [
                        (
                            ot,
                            args.option_underlying,
                            args.option_timespan,
                            args.option_from,
                            args.option_to,
                        )
                        for ot in tickers
                    ],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )
            if bad:
                return bad

        if tickers and args.fetch_quotes:
            bad = _first_bad(
                parallel_map(
                    _option_quotes_task,
                    [
                        (ot, args.option_underlying, args.quote_window_gte, args.quote_window_lt)
                        for ot in tickers
                    ],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )
            if bad:
                return bad

        if tickers and args.fetch_trades:
            bad = _first_bad(
                parallel_map(
                    _option_trades_task,
                    [
                        (ot, args.option_underlying, args.quote_window_gte, args.quote_window_lt)
                        for ot in tickers
                    ],
                    max_workers=args.jobs,
                    backend=args.parallel_backend,
                )
            )
            if bad:
                return bad
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--skip-stocks", action="store_true")
    p.add_argument("--skip-contracts", action="store_true")
    p.add_argument("--contracts-only", action="store_true")
    p.add_argument("--stock-1d-duration", default="2 Y")
    p.add_argument("--stock-1d-slug", default="2y")
    p.add_argument("--stock-1m-duration", default="10 D")
    p.add_argument("--stock-1m-slug", default="10d")
    p.add_argument("--expiration-date", default=None)
    p.add_argument("--option-tickers", default="")
    p.add_argument("--option-underlying", default="SPY")
    p.add_argument("--option-from", default="2026-01-01")
    p.add_argument("--option-to", default="2026-04-01")
    p.add_argument("--option-timespan", default="day", choices=("day", "minute", "hour"))
    p.add_argument("--skip-option-bars", action="store_true")
    p.add_argument("--fetch-quotes", action="store_true")
    p.add_argument("--fetch-trades", action="store_true")
    p.add_argument("--quote-window-gte", default="2026-03-20T13:30:00Z")
    p.add_argument("--quote-window-lt", default="2026-03-20T20:00:00Z")
    p.add_argument("--mtf", action="store_true")
    p.add_argument("--higher-tfs", default="1W,1M")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--parallel-backend", default="process", choices=("serial", "thread", "process", "ray"))
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return IngestionPipeline().run(args)


def _stock_task(task: tuple[str, str, str, str, str]) -> int:
    symbol, stock_1d_duration, stock_1d_slug, stock_1m_duration, stock_1m_slug = task
    write_ibkr_stock_parquet(symbol, duration=stock_1d_duration, bar_size="1 day", duration_slug=stock_1d_slug, interval_dir="1d")
    write_ibkr_stock_parquet(symbol, duration=stock_1m_duration, bar_size="1 min", duration_slug=stock_1m_slug, interval_dir="1m")
    return 0


def _contract_task(task: tuple[str, str | None]) -> int:
    symbol, expiration_date = task
    kw = {"limit": 1000}
    if expiration_date:
        kw["expiration_date"] = expiration_date
    write_massive_option_contracts_parquet(symbol, **kw)
    return 0


def _option_bars_task(task: tuple[str, str, str, str, str]) -> int:
    ot, underlying, timespan, opt_from, opt_to = task
    write_massive_option_bars_parquet(
        ot,
        underlying_for_path=underlying,
        multiplier=1,
        timespan=timespan,
        from_date=opt_from,
        to_date=opt_to,
    )
    return 0


def _option_quotes_task(task: tuple[str, str, str, str]) -> int:
    ot, underlying, ts_gte, ts_lt = task
    write_massive_option_quotes_parquet(ot, underlying_for_path=underlying, ts_gte=ts_gte, ts_lt=ts_lt)
    return 0


def _option_trades_task(task: tuple[str, str, str, str]) -> int:
    ot, underlying, ts_gte, ts_lt = task
    write_massive_option_trades_parquet(ot, underlying_for_path=underlying, ts_gte=ts_gte, ts_lt=ts_lt)
    return 0


def _first_bad(results: list[int]) -> int:
    return next((int(rc) for rc in results if int(rc) != 0), 0)
