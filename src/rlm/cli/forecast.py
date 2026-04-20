"""``rlm forecast`` — run the end-to-end forecast pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rlm.core.services.forecast_service import ForecastRequest, ForecastService
from rlm.core.pipeline import FullRLMConfig
from rlm.roee.engine import ROEEConfig
from rlm.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_SYMBOL = "SPY"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm forecast",
        description="Run factor + regime + ROEE forecast pipeline.",
    )
    p.add_argument("--symbol", default=_DEFAULT_SYMBOL, help="Ticker symbol (default: SPY)")
    p.add_argument("--bars", default=None, help="Path to bars CSV (default: data/raw/bars_SYMBOL.csv)")
    p.add_argument("--chain", default=None, help="Path to option chain CSV (optional)")
    p.add_argument("--out", default=None, help="Output CSV path (default: data/processed/forecast_features_SYMBOL.csv)")
    p.add_argument("--use-hmm", action="store_true", help="Use HMM regime model (default)")
    p.add_argument("--hmm-states", type=int, default=6)
    p.add_argument("--use-markov", action="store_true", help="Use Markov-switching model")
    p.add_argument("--markov-states", type=int, default=3)
    p.add_argument("--probabilistic", action="store_true")
    p.add_argument("--model-path", default=None, help="Quantile model artifact JSON")
    p.add_argument("--no-kronos", action="store_true", help="Disable Kronos overlay")
    p.add_argument("--no-vix", action="store_true", help="Skip VIX/VVIX attachment")
    p.add_argument("--run-backtest", action="store_true", help="Also run BacktestEngine (requires --chain)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")

    sym = args.symbol.upper().strip()
    root = Path(__file__).resolve().parents[4]

    bars_path = Path(args.bars) if args.bars else root / f"data/raw/bars_{sym}.csv"
    if not bars_path.is_file():
        raise SystemExit(
            f"Bars file not found: {bars_path}\n"
            f"Fetch data first: rlm ingest --symbol {sym}"
        )

    log.info("Loading bars: %s", bars_path)
    bars_df = pd.read_csv(bars_path, parse_dates=["timestamp"])
    bars_df = bars_df.sort_values("timestamp").set_index("timestamp")

    chain_df: pd.DataFrame | None = None
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
        run_backtest=args.run_backtest,
    )

    req = ForecastRequest(symbol=sym, bars_df=bars_df, option_chain_df=chain_df, config=cfg)
    log.info("Running forecast pipeline for %s (regime=%s, kronos=%s)", sym, regime_model, cfg.use_kronos)
    result = ForecastService().run(req)

    out_cols = [c for c in [
        "close", "S_D", "S_V", "S_L", "S_G",
        "b_m", "b_sigma", "mu", "sigma",
        "mean_price", "lower_1s", "upper_1s", "lower_2s", "upper_2s",
        "realized_vol", "forecast_source",
        "hmm_state", "hmm_state_label",
        "markov_state", "markov_state_label",
        "kronos_confidence", "kronos_regime_agreement",
        "kronos_predicted_regime", "kronos_transition_flag",
        "kronos_forecast_return", "kronos_forecast_vol",
    ] if c in result.forecast_df.columns]

    out_path = (
        Path(args.out) if args.out
        else root / f"data/processed/forecast_features_{sym}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.forecast_df.to_csv(out_path)
    log.info("Wrote %s", out_path)

    print(result.forecast_df[out_cols].tail(10).to_string())

    if args.run_backtest and result.backtest_metrics:
        print("\nBacktest metrics:")
        for k, v in result.backtest_metrics.items():
            print(f"  {k}: {v}")
