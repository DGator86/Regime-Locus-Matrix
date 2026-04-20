"""``rlm backtest`` — run backtests and walk-forward analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rlm.cli.common import (
    add_backend_arg,
    add_data_root_arg,
    add_pipeline_args,
    add_profile_args,
    build_pipeline_config,
    normalize_symbol,
)
from rlm.core.run_manifest import RunManifest, write_run_manifest
from rlm.core.services.backtest_service import BacktestRequest, BacktestService
from rlm.data.paths import get_processed_data_dir
from rlm.data.readers import load_bars, load_option_chain
from rlm.data.synthetic import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.utils.logging import get_run_logger
from rlm.utils.run_id import generate_run_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rlm backtest", description="Run RLM backtest (optionally with walk-forward).")
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--bars", default=None, help="Path to bars CSV")
    p.add_argument("--chain", default=None, help="Path to option chain CSV")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic demo data")
    p.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--out-dir", default=None)
    add_pipeline_args(p)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)
    run_id = generate_run_id("backtest")
    log = get_run_logger(__name__, run_id=run_id, command="backtest", symbol=sym, backend=args.backend, profile=args.profile)

    bars_df: pd.DataFrame
    chain_df: pd.DataFrame | None = None
    if args.synthetic:
        bars_df = synthetic_bars_demo(end=pd.Timestamp.today(), periods=220)
        chain_df = synthetic_option_chain_from_bars(bars_df)
    else:
        bars_df = load_bars(sym, bars_path=args.bars, data_root=args.data_root, backend=args.backend)
        chain_df = load_option_chain(sym, chain_path=args.chain, data_root=args.data_root, backend=args.backend)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else get_processed_data_dir(args.data_root)
    cfg = build_pipeline_config(args, sym)

    svc = BacktestService()
    req = BacktestRequest(symbol=sym, bars_df=bars_df, option_chain_df=chain_df, config=cfg, walkforward=args.walkforward, write_outputs=True, out_dir=out_dir, initial_capital=args.initial_capital)
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

    manifest = RunManifest(
        run_id=run_id,
        command="backtest",
        symbol=sym,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        profile=args.profile,
        config_summary={"regime_model": cfg.regime_model, "walkforward": args.walkforward},
        input_paths={"bars": args.bars or "auto", "chain": args.chain or "auto"},
        output_paths={"trades_csv": str(arts.trades_csv) if arts.trades_csv else "", "equity_csv": str(arts.equity_csv) if arts.equity_csv else ""},
        metrics=summary.get("metrics", {}),
    )
    manifest_path = write_run_manifest(manifest, data_root=args.data_root)
    log.info("backtest complete", extra={"stage": "complete", "success": True})
    print(f"Run manifest: {manifest_path}")
