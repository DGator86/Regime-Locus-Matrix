"""``rlm backtest`` — run backtests and walk-forward analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlm.cli.common import add_data_root_arg, add_pipeline_args, build_pipeline_config, normalize_symbol
from rlm.cli.io import resolve_bars_path, resolve_chain_path, resolve_output_path
from rlm.core.services.backtest_service import BacktestRequest, BacktestService
from rlm.data.paths import get_processed_data_dir
from rlm.data.readers import load_bars, load_option_chain
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm backtest",
        description="Run RLM backtest (optionally with walk-forward).",
    )
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--bars", default=None, help="Path to bars CSV")
    p.add_argument("--chain", default=None, help="Path to option chain CSV")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic demo data (no real files needed)")
    p.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--out-dir", default=None, help="Output directory (default: <data-root>/processed/)")
    add_pipeline_args(p)
    add_data_root_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)

    bars_df: pd.DataFrame
    chain_df: pd.DataFrame | None = None

    if args.synthetic:
        from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
        bars_df = synthetic_bars_demo(end=pd.Timestamp.today(), periods=220)
        chain_df = synthetic_option_chain_from_bars(bars_df)
        log.info("backtest using synthetic data  symbol=%s bars=%d", sym, len(bars_df))
    else:
        bars_path = resolve_bars_path(sym, args.bars, args.data_root)
        log.info("backtest start  symbol=%s bars=%s", sym, bars_path)
        bars_df = load_bars(sym, bars_path=bars_path)

        chain_path = resolve_chain_path(sym, args.chain, args.data_root)
        chain_df = load_option_chain(sym, chain_path=chain_path) if chain_path else None

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else get_processed_data_dir(args.data_root)
    )

    cfg = build_pipeline_config(args, sym)

    svc = BacktestService()
    req = BacktestRequest(
        symbol=sym,
        bars_df=bars_df,
        option_chain_df=chain_df,
        config=cfg,
        walkforward=args.walkforward,
        write_outputs=True,
        out_dir=out_dir,
        initial_capital=args.initial_capital,
    )

    result = svc.run(req)
    arts = svc.write_outputs(req, result)
    summary = svc.summarize(result)

    if summary.get("metrics"):
        print("\nBacktest metrics:")
        for k, v in summary["metrics"].items():
            print(f"  {k}: {v}")

    if arts.trades_csv:
        print(f"\nTrades:       {arts.trades_csv}  ({arts.trades_rows} rows)")
    if arts.equity_csv:
        print(f"Equity curve: {arts.equity_csv}  ({arts.equity_rows} rows)")
