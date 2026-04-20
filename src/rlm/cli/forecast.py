"""``rlm forecast`` — run the end-to-end forecast pipeline."""

from __future__ import annotations

import argparse

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_pipeline_args, add_profile_args, build_pipeline_config, normalize_symbol
from rlm.cli.io import resolve_bars_path, resolve_chain_path, resolve_output_path
from rlm.core.services.forecast_service import ForecastRequest, ForecastService
from rlm.data.readers import load_bars, load_option_chain
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm forecast",
        description="Run factor + regime + ROEE forecast pipeline.",
    )
    p.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    p.add_argument("--bars", default=None, help="Path to bars CSV")
    p.add_argument("--chain", default=None, help="Path to option chain CSV (optional)")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--run-backtest", action="store_true", help="Also run BacktestEngine (requires --chain)")
    add_pipeline_args(p)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)

    bars_path = resolve_bars_path(sym, args.bars, args.data_root)
    log.info("forecast start  symbol=%s bars=%s", sym, bars_path)

    bars_df = load_bars(sym, bars_path=bars_path, backend=args.backend)

    chain_path = resolve_chain_path(sym, args.chain, args.data_root)
    chain_df = load_option_chain(sym, chain_path=chain_path, backend=args.backend) if chain_path else None

    cfg = build_pipeline_config(args, sym)
    cfg.run_backtest = args.run_backtest

    out_path = resolve_output_path("forecast_features", sym, args.out, args.data_root)

    svc = ForecastService()
    req = ForecastRequest(
        symbol=sym,
        bars_df=bars_df,
        option_chain_df=chain_df,
        config=cfg,
        out_path=out_path,
        write_output=True,
    )

    result = svc.run(req)
    arts = svc.write_outputs(req, result)
    summary = svc.summarize(result)

    out_cols = svc.output_columns(result)
    print(result.forecast_df[out_cols].tail(req.tail_rows).to_string())

    if arts.forecast_csv:
        print(f"\nWrote {arts.forecast_csv}")

    if summary.get("action"):
        print(f"\nLatest ROEE decision: action={summary['action']}  strategy={summary['strategy']}  size={summary['size_fraction']}")

    if args.run_backtest and summary.get("backtest"):
        print("\nBacktest metrics:")
        for k, v in summary["backtest"].items():
            print(f"  {k}: {v}")
