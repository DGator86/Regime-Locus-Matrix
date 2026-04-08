from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.backtest.engine import BacktestEngine
from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import (
    DEFAULT_SYMBOL,
    backtest_equity_filename,
    backtest_trades_filename,
    rel_bars_csv,
    rel_option_chain_csv,
)
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
    HybridProbabilisticForecastPipeline,
)
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.roee.pipeline import ROEEConfig
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RLM backtest. Use --today or --only-date to evaluate a single day "
        "(features still use full history for warm-up)."
    )
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    parser.add_argument(
        "--use-markov", action="store_true", help="Use Markov-switching regime model."
    )
    parser.add_argument("--markov-states", type=int, default=3)
    parser.add_argument(
        "--probabilistic", action="store_true", help="Use probabilistic forecast output."
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Optional quantile model artifact JSON."
    )
    parser.add_argument(
        "--dynamic-sizing", action="store_true", help="Enable Kelly/vol-target sizing."
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly cap to use when dynamic sizing is enabled.",
    )
    parser.add_argument(
        "--no-regime-adjusted-kelly",
        action="store_false",
        dest="regime_adjusted_kelly",
        help="Disable latent-regime Kelly adjustment (enabled by default).",
    )
    parser.add_argument(
        "--vault-uncertainty-threshold",
        type=float,
        default=0.03,
        help="Cut size when forecast 5th-95th range exceeds this return-width threshold.",
    )
    parser.add_argument(
        "--vault-size-multiplier",
        type=float,
        default=0.5,
        help="Position-size multiplier to apply when the Vault rule triggers.",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Restrict the engine to the current calendar date (local).",
    )
    parser.add_argument(
        "--only-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Restrict the engine to this calendar date (after full pipeline warm-up).",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Underlying for default --bars/--chain paths and engine (default {DEFAULT_SYMBOL}).",
    )
    parser.add_argument(
        "--bars",
        type=str,
        default=None,
        help="CSV with timestamp column (default: data/raw/bars_{SYMBOL}.csv).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Option chain CSV (default: data/raw/option_chain_{SYMBOL}.csv).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore --bars/--chain files and build in-memory demo data (needed if CSVs absent).",
    )
    parser.add_argument(
        "--warmup-days", type=int, default=220, help="Synthetic history length (days)."
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip yfinance VIX/VVIX when enriching bars from the option chain.",
    )
    return parser.parse_args()


def _load_or_synthetic_bars(
    path: str, *, synthetic: bool, end: pd.Timestamp, warmup_days: int
) -> pd.DataFrame:
    if synthetic:
        return synthetic_bars_demo(end, periods=max(warmup_days, 120))
    p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(f"Bars file not found: {p}. Pass --synthetic or create the CSV.")
    bars = pd.read_csv(p, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")
    return bars


def _load_or_synthetic_chain(
    path: str, bars: pd.DataFrame, *, synthetic: bool, underlying: str
) -> pd.DataFrame:
    if synthetic:
        return synthetic_option_chain_from_bars(bars, underlying=underlying)
    p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(f"Chain file not found: {p}. Pass --synthetic or create the CSV.")
    return pd.read_csv(p, parse_dates=["timestamp", "expiry"])


def main() -> None:
    args = parse_args()
    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    chain_rel = args.chain or rel_option_chain_csv(sym)

    if args.today and args.only_date:
        raise SystemExit("Use either --today or --only-date, not both.")

    if args.today:
        only_ts = pd.Timestamp(date.today())
    elif args.only_date:
        only_ts = pd.Timestamp(datetime.strptime(args.only_date, "%Y-%m-%d").date())
    else:
        only_ts = None

    end = only_ts if only_ts is not None else pd.Timestamp.today().normalize()
    bars = _load_or_synthetic_bars(
        bars_rel,
        synthetic=args.synthetic,
        end=end,
        warmup_days=args.warmup_days,
    )
    chain = _load_or_synthetic_chain(
        chain_rel,
        bars,
        synthetic=args.synthetic,
        underlying=sym,
    )

    bars = prepare_bars_for_factors(bars, chain, underlying=sym, attach_vix=not args.no_vix)

    features = FactorPipeline().run(bars)
    fc = ForecastConfig(
        drift_gamma_alpha=0.65,
        sigma_floor=1e-4,
        direction_neutral_threshold=0.3,
    )
    if args.use_hmm and args.probabilistic:
        features = HybridProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
            model_path=args.model_path,
        ).run(features)
    elif args.use_hmm:
        features = HybridForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
        ).run(features)
    elif args.use_markov and args.probabilistic:
        features = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
            model_path=args.model_path,
        ).run(features)
    elif args.use_markov:
        features = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
        ).run(features)
    elif args.probabilistic:
        features = ProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            model_path=args.model_path,
        ).run(features)
    else:
        features = ForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
        ).run(features)

    if only_ts is not None:
        d = only_ts.normalize()
        feat_mask = features.index.normalize() == d
        features_run = features.loc[feat_mask]
        chain_run = chain.loc[pd.to_datetime(chain["timestamp"]).dt.normalize() == d].copy()
        if features_run.empty:
            raise SystemExit(f"No bar rows on {d.date()} after warm-up. Check data range.")
        if chain_run.empty:
            raise SystemExit(f"No option chain rows on {d.date()}. Check chain coverage.")
        print(f"Single-day backtest as-of {d.date()} (1 bar, warm-up used {len(features)} days).")
    else:
        features_run = features
        chain_run = chain
        print(f"Full backtest: {len(features_run)} bars.")

    engine = BacktestEngine(
        initial_capital=100_000.0,
        contract_multiplier=100,
        strike_increment=5.0,
        underlying_symbol=sym,
        quantity_per_trade=1,
        roee_config=ROEEConfig(
            use_dynamic_sizing=args.dynamic_sizing,
            max_kelly_fraction=args.kelly_fraction,
            regime_adjusted_kelly=args.regime_adjusted_kelly,
            vault_uncertainty_threshold=args.vault_uncertainty_threshold,
            vault_size_multiplier=args.vault_size_multiplier,
        ),
    )

    equity_frame, trades_frame, summary = engine.run(features_run, chain_run)

    print("Backtest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / backtest_equity_filename(sym)
    trades_path = out_dir / backtest_trades_filename(sym)
    equity_frame.to_csv(equity_path)
    trades_frame.to_csv(trades_path, index=False)
    print(f"Wrote {equity_path.relative_to(ROOT)} and {trades_path.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()