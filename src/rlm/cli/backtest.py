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
    p = argparse.ArgumentParser(
        prog="rlm backtest",
        description="Run RLM backtest (optionally with walk-forward).",
    )
    sym_grp = p.add_mutually_exclusive_group()
    sym_grp.add_argument("--symbol", default=None, help="Single symbol (e.g. SPY)")
    sym_grp.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols, e.g. SPY,AAPL,QQQ",
    )
    sym_grp.add_argument(
        "--universe",
        action="store_true",
        help="Run on the full LIQUID_UNIVERSE (Mag7 + SPY + QQQ)",
    )
    p.add_argument("--bars", default=None, help="Path to bars CSV (single-symbol only)")
    p.add_argument("--chain", default=None, help="Path to option chain CSV (single-symbol only)")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic demo data")
    p.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--out-dir", default=None)
    add_pipeline_args(p)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    from rlm.data.liquidity_universe import LIQUID_UNIVERSE

    if args.universe:
        return list(LIQUID_UNIVERSE)
    if args.symbols:
        return [
            s.strip().upper()
            for s in args.symbols.replace(";", ",").split(",")
            if s.strip()
        ]
    if args.symbol:
        return [normalize_symbol(args.symbol)]
    return ["SPY"]


def _run_one(
    sym: str,
    args: argparse.Namespace,
    out_dir: Path,
    run_id: str,
) -> None:
    log = get_run_logger(
        __name__,
        run_id=run_id,
        command="backtest",
        symbol=sym,
        backend=args.backend,
        profile=args.profile,
    )

    bars_df: pd.DataFrame
    chain_df: pd.DataFrame | None = None

    if args.synthetic:
        bars_df = synthetic_bars_demo(end=pd.Timestamp.today(), periods=220)
        chain_df = synthetic_option_chain_from_bars(bars_df)
    else:
        try:
            bars_df = load_bars(
                sym,
                bars_path=args.bars,
                data_root=args.data_root,
                backend=args.backend,
            )
            chain_df = load_option_chain(
                sym,
                chain_path=args.chain,
                data_root=args.data_root,
                backend=args.backend,
            )
        except FileNotFoundError as exc:
            print(f"  {sym}: skipping — {exc}")
            log.warning("no data for symbol, skipping", extra={"symbol": sym})
            return

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
        print(f"\n{sym} metrics:")
        for k, v in summary["metrics"].items():
            print(f"  {k}: {v}")
    if arts.trades_csv:
        print(f"  Trades:       {arts.trades_csv}  ({arts.trades_rows} rows)")
    if arts.equity_csv:
        print(f"  Equity curve: {arts.equity_csv}  ({arts.equity_rows} rows)")

    manifest = RunManifest(
        run_id=run_id,
        command="backtest",
        symbol=sym,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        profile=args.profile,
        config_summary={"regime_model": cfg.regime_model, "walkforward": args.walkforward},
        input_paths={"bars": args.bars or "auto", "chain": args.chain or "auto"},
        output_paths={
            "trades_csv": str(arts.trades_csv) if arts.trades_csv else "",
            "equity_csv": str(arts.equity_csv) if arts.equity_csv else "",
        },
        metrics=summary.get("metrics", {}),
    )
    write_run_manifest(manifest, data_root=args.data_root)
    log.info("backtest complete", extra={"stage": "complete", "success": True})


def main() -> None:
    args = _parse_args()
    symbols = _resolve_symbols(args)
    batch_run_id = generate_run_id("backtest")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else get_processed_data_dir(args.data_root)
    )

    if len(symbols) > 1:
        print(
            f"Running backtest {'+ walk-forward ' if args.walkforward else ''}on "
            f"{len(symbols)} symbols: {', '.join(symbols)}"
        )

    skipped = 0
    for idx, sym in enumerate(symbols, start=1):
        run_id = (
            batch_run_id
            if len(symbols) == 1
            else f"{batch_run_id}-{idx:02d}-{sym}"
        )
        try:
            _run_one(sym, args, out_dir, run_id)
        except Exception as exc:
            print(f"  {sym}: ERROR — {exc}")
            skipped += 1

    if len(symbols) > 1:
        done = len(symbols) - skipped
        print(f"\nDone: {done}/{len(symbols)} symbols completed.")
        if args.walkforward:
            print(
                f"Walk-forward summaries written to: {out_dir}/walkforward_summary_*.csv"
            )
