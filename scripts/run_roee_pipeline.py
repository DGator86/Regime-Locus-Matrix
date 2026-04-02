from __future__ import annotations

import argparse

import pandas as pd

from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.pipeline import ForecastPipeline, HybridForecastPipeline
from rlm.roee.pipeline import ROEEConfig, apply_roee_policy
from rlm.scoring.state_matrix import classify_state_matrix
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv("data/raw/sample_bars.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    factor_df = FactorPipeline().run(df)
    if args.use_hmm:
        forecast_df = HybridForecastPipeline(
            config=ForecastConfig(
                drift_gamma_alpha=0.65,
                sigma_floor=1e-4,
                direction_neutral_threshold=0.3,
            ),
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
        ).run(factor_df)
    else:
        forecast_df = ForecastPipeline(
            config=ForecastConfig(
                drift_gamma_alpha=0.65,
                sigma_floor=1e-4,
                direction_neutral_threshold=0.3,
            ),
            move_window=100,
            vol_window=100,
        ).run(factor_df)

    state_df = classify_state_matrix(forecast_df)
    policy_df = apply_roee_policy(state_df, strike_increment=5.0, config=ROEEConfig())

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
        "roee_action",
        "roee_strategy",
        "roee_size_fraction",
        "roee_leg_count",
        "hmm_confidence",
        "hmm_size_mult",
        "hmm_trade_allowed",
    ]
    print(policy_df[[c for c in cols if c in policy_df.columns]].tail(15))
    policy_df.to_csv("data/processed/roee_policy_output.csv")


if __name__ == "__main__":
    main()
