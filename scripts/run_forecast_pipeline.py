from __future__ import annotations

import pandas as pd

from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.types.forecast import ForecastConfig


def main() -> None:
    df = pd.read_csv("data/raw/sample_bars.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    factor_pipeline = FactorPipeline()
    forecast_pipeline = ForecastPipeline(
        config=ForecastConfig(
            drift_gamma_alpha=0.65,
            sigma_floor=1e-4,
            direction_neutral_threshold=0.3,
        ),
        move_window=100,
        vol_window=100,
    )

    factors = factor_pipeline.run(df)
    forecast = forecast_pipeline.run(factors)

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

    forecast.to_csv("data/processed/forecast_features.csv")
    print(forecast[out_cols].tail(10))


if __name__ == "__main__":
    main()
