import numpy as np
import pandas as pd

from rlm.factors import FactorPipeline
from rlm.forecasting.engines import ForecastPipeline
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.types.forecast import ForecastConfig


def make_sample_bars(n: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    trend = np.linspace(5000, 5100, n)
    wave = np.sin(np.arange(n) / 6.0) * 8.0

    df = pd.DataFrame(
        {
            "open": trend + wave - 1.0,
            "high": trend + wave + 2.0,
            "low": trend + wave - 2.0,
            "close": trend + wave,
            "volume": 100000 + (np.arange(n) % 10) * 4000,
            "vwap": trend + wave * 0.2,
            "anchored_vwap": trend - 3.0,
            "buy_volume": 60000 + (np.arange(n) % 5) * 2000,
            "sell_volume": 45000 + (np.arange(n) % 7) * 1500,
            "advancers": 1700 + (np.arange(n) % 20) * 8,
            "decliners": 1300 + (np.arange(n) % 18) * 9,
            "index_return": pd.Series(trend).pct_change(10).fillna(0.0).values,
            "vix": 17 + (np.arange(n) % 15) * 0.4,
            "vvix": 88 + (np.arange(n) % 10) * 1.2,
            "bid_ask_spread": 0.45 + (np.arange(n) % 6) * 0.03,
            "order_book_depth": 2500 + (np.arange(n) % 9) * 120,
            "gex": np.sin(np.arange(n) / 11.0),
            "vanna": np.cos(np.arange(n) / 13.0),
            "charm": np.sin(np.arange(n) / 9.0),
            "put_call_skew": 0.02 + (np.arange(n) % 4) * 0.006,
            "iv_rank": 0.4 + (np.arange(n) % 12) * 0.025,
            "term_structure_ratio": 0.96 + (np.arange(n) % 5) * 0.025,
            "dealer_position_proxy": np.sin(np.arange(n) / 17.0) * 0.2,
        },
        index=idx,
    )
    return df


def test_forecast_pipeline_outputs_distribution_columns() -> None:
    df = make_sample_bars()

    factor_out = FactorPipeline().run(df)
    forecast_out = ForecastPipeline(
        config=ForecastConfig(sigma_floor=1e-4),
        move_window=50,
        vol_window=50,
    ).run(factor_out)

    required = [
        "b_m",
        "b_sigma",
        "mu",
        "sigma",
        "mean_price",
        "lower_1s",
        "upper_1s",
        "lower_2s",
        "upper_2s",
        "forecast_return",
        "forecast_return_lower",
        "forecast_return_median",
        "forecast_return_upper",
        "forecast_uncertainty",
        "realized_vol",
    ]
    for col in required:
        assert col in forecast_out.columns


def test_sigma_respects_floor() -> None:
    df = make_sample_bars()

    factor_out = FactorPipeline().run(df)
    forecast_out = ForecastPipeline(
        config=ForecastConfig(sigma_floor=0.001),
        move_window=50,
        vol_window=50,
    ).run(factor_out)

    sigma = forecast_out["sigma"].dropna()
    assert len(sigma) > 0
    assert (sigma >= 0.001).all()


def test_band_ordering_is_valid() -> None:
    df = make_sample_bars()

    factor_out = FactorPipeline().run(df)
    forecast_out = ForecastPipeline(
        config=ForecastConfig(sigma_floor=1e-4),
        move_window=50,
        vol_window=50,
    ).run(factor_out)

    valid = forecast_out.dropna(
        subset=["lower_2s", "lower_1s", "mean_price", "upper_1s", "upper_2s"]
    )

    assert (valid["lower_2s"] <= valid["lower_1s"]).all()
    assert (valid["lower_1s"] <= valid["mean_price"]).all()
    assert (valid["mean_price"] <= valid["upper_1s"]).all()
    assert (valid["upper_1s"] <= valid["upper_2s"]).all()


def test_probabilistic_pipeline_fallback_emits_ordered_quantiles() -> None:
    df = make_sample_bars()

    factor_out = FactorPipeline().run(df)
    forecast_out = ProbabilisticForecastPipeline(
        config=ForecastConfig(sigma_floor=1e-4),
        move_window=50,
        vol_window=50,
    ).run(factor_out)

    valid = forecast_out.dropna(
        subset=[
            "forecast_return_lower",
            "forecast_return_median",
            "forecast_return_upper",
            "forecast_uncertainty",
            "forecast_source",
        ]
    )

    assert not valid.empty
    assert (valid["forecast_return_lower"] <= valid["forecast_return_median"]).all()
    assert (valid["forecast_return_median"] <= valid["forecast_return_upper"]).all()
    assert (valid["forecast_uncertainty"] >= 0.0).all()
    assert set(valid["forecast_source"]) == {"distribution_fallback"}
