from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_option_chain_csv, rel_roee_policy_csv
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
from rlm.roee.pipeline import ROEEConfig, apply_roee_policy
from rlm.scoring.state_matrix import classify_state_matrix
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factors → forecast → state matrix → ROEE policy (default bars: data/raw/bars_{SYMBOL}.csv)."
    )
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    parser.add_argument("--use-markov", action="store_true", help="Use Markov-switching regime overlay.")
    parser.add_argument("--markov-states", type=int, default=3, help="Number of Markov regimes.")
    parser.add_argument("--probabilistic", action="store_true", help="Use probabilistic forecast output.")
    parser.add_argument("--model-path", default=None, help="Optional quantile model artifact JSON.")
    parser.add_argument("--dynamic-sizing", action="store_true", help="Enable Kelly/vol-target sizing.")
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
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Ticker for default paths (default {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--bars",
        default=None,
        help="Bars CSV relative to repo root (default: data/raw/bars_{SYMBOL}.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV relative to repo root (default: data/processed/roee_policy_{SYMBOL}.csv)",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="Option chain CSV for enrichment (default: data/raw/option_chain_{SYMBOL}.csv if present)",
    )
    parser.add_argument("--no-vix", action="store_true", help="Skip yfinance VIX/VVIX.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    out_rel = args.out or rel_roee_policy_csv(sym)
    bars_path = ROOT / bars_rel
    if not bars_path.is_file():
        raise SystemExit(
            f"Bars file not found: {bars_path}\n"
            "See: python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol "
            f"{sym} --start YYYY-MM-DD"
        )

    df = pd.read_csv(bars_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    chain_rel = args.chain or rel_option_chain_csv(sym)
    cp = ROOT / chain_rel
    opch = pd.read_csv(cp, parse_dates=["timestamp", "expiry"]) if cp.is_file() else None
    df = prepare_bars_for_factors(df, opch, underlying=sym, attach_vix=not args.no_vix)

    factor_df = FactorPipeline().run(df)
    fc = ForecastConfig(
        drift_gamma_alpha=0.65,
        sigma_floor=1e-4,
        direction_neutral_threshold=0.3,
    )
    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    if args.use_hmm and args.probabilistic:
        forecast_df = HybridProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
            model_path=args.model_path,
        ).run(factor_df)
    elif args.use_markov and args.probabilistic:
        forecast_df = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
            model_path=args.model_path,
        ).run(factor_df)
    elif args.use_hmm:
        forecast_df = HybridForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
        ).run(factor_df)
    elif args.use_markov:
        forecast_df = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
        ).run(factor_df)
    elif args.probabilistic:
        forecast_df = ProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            model_path=args.model_path,
        ).run(factor_df)
    else:
        forecast_df = ForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
        ).run(factor_df)

    state_df = classify_state_matrix(forecast_df)
    policy_df = apply_roee_policy(
        state_df,
        strike_increment=5.0,
        config=ROEEConfig(
            use_dynamic_sizing=args.dynamic_sizing,
            vault_uncertainty_threshold=args.vault_uncertainty_threshold,
            vault_size_multiplier=args.vault_size_multiplier,
        ),
    )

    cols = [
        "close",
        "S_D",
        "S_V",
        "S_L",
        "S_G",
        "direction_regime",
        "volatility_regime",
        "liquidity_regime",
        "dealer_flow_regime",
        "regime_key",
        "sigma",
        "forecast_return",
        "forecast_return_lower",
        "forecast_return_median",
        "forecast_return_upper",
        "forecast_uncertainty",
        "realized_vol",
        "roee_action",
        "roee_strategy",
        "roee_size_fraction",
        "vault_triggered",
        "vault_size_multiplier",
        "vault_uncertainty_threshold",
        "roee_leg_count",
        "hmm_confidence",
        "hmm_size_mult",
        "hmm_trade_allowed",
        "markov_state",
        "markov_state_label",
        "markov_confidence",
    ]
    print(policy_df[[c for c in cols if c in policy_df.columns]].tail(15))
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    policy_df.to_csv(out_path)
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
