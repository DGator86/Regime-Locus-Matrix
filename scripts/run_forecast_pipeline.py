from __future__ import annotations

import argparse

import pandas as pd

from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.pipeline import ForecastPipeline, HybridForecastPipeline
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

    factor_pipeline = FactorPipeline()

    factors = factor_pipeline.run(df)
    if args.use_hmm:
        forecast = HybridForecastPipeline(
            config=ForecastConfig(
                drift_gamma_alpha=0.65,
                sigma_floor=1e-4,
                direction_neutral_threshold=0.3,
            ),
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
        ).run(factors)
    else:
        forecast = ForecastPipeline(
            config=ForecastConfig(
                drift_gamma_alpha=0.65,
                sigma_floor=1e-4,
                direction_neutral_threshold=0.3,
            ),
            move_window=100,
            vol_window=100,
        ).run(factors)

    out_cols = [
        "close",
        "S_D",
        "S_V",
        "S_L",
        "S_G",
        "b_m",
        "b_sigma",
        "mu",
        "sigma",
        "mean_price",
        "lower_1s",
        "upper_1s",
        "lower_2s",
        "upper_2s",
    ]
    if args.use_hmm:
        out_cols.extend(["hmm_state", "hmm_state_label"])

    forecast.to_csv("data/processed/forecast_features.csv")
    print(forecast[out_cols].tail(10))


if __name__ == "__main__":
    main()
