"""``rlm backtest`` — run backtests and walk-forward analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlm.core.services.backtest_service import BacktestRequest, BacktestService
from rlm.core.pipeline import FullRLMConfig
from rlm.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_SYMBOL = "SPY"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm backtest",
        description="Run RLM backtest (optionally with walk-forward).",
    )
    p.add_argument("--symbol", default=_DEFAULT_SYMBOL)
    p.add_argument("--bars", default=None, help="Path to bars CSV")
    p.add_argument("--chain", default=None, help="Path to option chain CSV")
    p.add_argument("--use-hmm", action="store_true")
    p.add_argument("--hmm-states", type=int, default=6)
    p.add_argument("--use-markov", action="store_true")
    p.add_argument("--markov-states", type=int, default=3)
    p.add_argument("--probabilistic", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--no-kronos", action="store_true")
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic demo data (no real data needed)")
    p.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--out-dir", default=None, help="Output directory for results (default: data/processed/)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")

    sym = args.symbol.upper().strip()
    root = Path(__file__).resolve().parents[4]
    out_dir = Path(args.out_dir) if args.out_dir else root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_df: pd.DataFrame | None = None
    chain_df: pd.DataFrame | None = None

    if args.synthetic:
        from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
        bars_df = synthetic_bars_demo(end=pd.Timestamp.today(), periods=220)
        chain_df = synthetic_option_chain_from_bars(bars_df)
        log.info("Using synthetic demo data (%d bars)", len(bars_df))
    else:
        bars_path = Path(args.bars) if args.bars else root / f"data/raw/bars_{sym}.csv"
        if not bars_path.is_file():
            raise SystemExit(
                f"Bars file not found: {bars_path}\n"
                f"Fetch data first: rlm ingest --symbol {sym}"
            )
        log.info("Loading bars: %s", bars_path)
        bars_df = pd.read_csv(bars_path, parse_dates=["timestamp"])
        bars_df = bars_df.sort_values("timestamp").set_index("timestamp")

        chain_path = Path(args.chain) if args.chain else root / f"data/raw/option_chain_{sym}.csv"
        if chain_path.is_file():
            log.info("Loading option chain: %s", chain_path)
            chain_df = pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"])

    regime_model = "none"
    if args.use_hmm or (not args.use_markov):
        regime_model = "hmm"
    if args.use_markov:
        regime_model = "markov"

    cfg = FullRLMConfig(
        symbol=sym,
        regime_model=regime_model,  # type: ignore[arg-type]
        hmm_states=args.hmm_states,
        markov_states=args.markov_states,
        probabilistic=args.probabilistic,
        probabilistic_model_path=args.model_path,
        use_kronos=not args.no_kronos,
        attach_vix=not args.no_vix,
        run_backtest=True,
        initial_capital=args.initial_capital,
    )

    req = BacktestRequest(
        symbol=sym,
        bars_df=bars_df,
        option_chain_df=chain_df,
        config=cfg,
        walkforward=args.walkforward,
        out_dir=out_dir,
    )

    log.info("Running backtest for %s (regime=%s, walkforward=%s)", sym, regime_model, args.walkforward)
    service = BacktestService()
    result = service.run(req)

    if result.backtest_metrics:
        print("\nBacktest metrics:")
        for k, v in result.backtest_metrics.items():
            print(f"  {k}: {v}")

    if result.backtest_trades is not None and not result.backtest_trades.empty:
        trades_path = out_dir / f"backtest_trades_{sym}.csv"
        result.backtest_trades.to_csv(trades_path)
        log.info("Wrote trades: %s", trades_path)

    if result.backtest_equity is not None and not result.backtest_equity.empty:
        equity_path = out_dir / f"backtest_equity_{sym}.csv"
        result.backtest_equity.to_csv(equity_path)
        log.info("Wrote equity curve: %s", equity_path)
