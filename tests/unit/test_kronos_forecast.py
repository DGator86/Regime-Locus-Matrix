"""Tests for KronosForecastPipeline and KronosFactorCalculator using mocks."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rlm.factors import KronosFactorCalculator
from rlm.forecasting.kronos_config import KronosConfig
from rlm.forecasting.kronos_forecast import KronosForecastPipeline


def _make_bars(n: int = 60) -> pd.DataFrame:
    """
    Generate a deterministic synthetic OHLCV DataFrame for testing.

    Creates `n` daily rows starting at 2024-01-01 using a fixed random seed so results are reproducible. The returned DataFrame contains the columns:
    - `timestamp`: daily timestamps starting 2024-01-01
    - `open`, `high`, `low`, `close`: synthetic prices
    - `volume`: integer trading volumes in the range [100000, 200000)

    Parameters:
        n (int): Number of rows (days) to generate.

    Returns:
        pd.DataFrame: Synthetic OHLCV data with columns `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
    """
    rng = np.random.RandomState(42)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close - rng.rand(n) * 0.2,
            "high": close + rng.rand(n) * 0.5,
            "low": close - rng.rand(n) * 0.5,
            "close": close,
            "volume": rng.randint(100_000, 200_000, size=n),
            "S_D": rng.randn(n) * 0.1,
            "S_V": rng.randn(n) * 0.1,
            "S_L": rng.randn(n) * 0.1,
            "S_G": rng.randn(n) * 0.1,
        }
    )


def _mock_predictor(sample_count: int = 5, pred_len: int = 3):
    """
    Create a MagicMock predictor whose `predict_paths` returns deterministic forecast paths.

    Parameters:
        sample_count (int): Number of sample paths to generate per call.
        pred_len (int): Number of future time steps per path.

    Returns:
        MagicMock: A mock object with a `predict_paths(df, future_timestamps=None)` callable that returns a NumPy array of shape (sample_count, pred_len, 6). Each step contains six deterministic features derived from the last `close` in `df`: [low, high, alternate_low, close, volume, volume * close].
    """
    mock = MagicMock()

    def _predict_paths(df, future_timestamps=None):
        """
        Create deterministic simulated forecast paths anchored to the last close price in `df`.

        Parameters:
            df (pd.DataFrame): Historical OHLCV bars; the last row's `close` value is used as the starting price.
            future_timestamps (optional): Ignored by this mock; present for API compatibility.

        Returns:
            np.ndarray: Array of shape (sample_count, pred_len, 6) with simulated future steps. Each step contains
            [low, high, open, close, volume, volume * price] constructed deterministically from the last close.
        """
        current_close = float(df["close"].iloc[-1])
        paths = np.empty((sample_count, pred_len, 6))
        for i in range(sample_count):
            drift = 0.003 * (i - sample_count // 2)
            for t in range(pred_len):
                c = current_close * (1 + drift * (t + 1))
                paths[i, t, :] = [c - 0.1, c + 0.4, c - 0.4, c, 150000, 150000 * c]
        return paths

    mock.predict_paths = _predict_paths
    return mock


# ── KronosForecastPipeline ───────────────────────────────────────────


class TestKronosForecastPipeline:
    def test_run_produces_forecast_columns(self):
        cfg = KronosConfig(sample_count=5, pred_len=3, max_context=60)
        pred = _mock_predictor(5, 3)
        pipeline = KronosForecastPipeline(config=cfg, predictor=pred, min_lookback=30)

        bars = _make_bars(60)
        result = pipeline.run(bars)

        assert "mu" in result.columns
        assert "sigma" in result.columns
        assert "mean_price" in result.columns
        assert "forecast_source" in result.columns
        assert (result["forecast_source"] == "kronos").all()

    def test_sigma_is_positive(self):
        cfg = KronosConfig(sample_count=5, pred_len=3, max_context=60)
        pred = _mock_predictor(5, 3)
        pipeline = KronosForecastPipeline(config=cfg, predictor=pred, min_lookback=30)

        bars = _make_bars(60)
        result = pipeline.run(bars)
        assert (result["sigma"] > 0).all()

    def test_band_columns_present(self):
        cfg = KronosConfig(sample_count=5, pred_len=3, max_context=60)
        pred = _mock_predictor(5, 3)
        pipeline = KronosForecastPipeline(config=cfg, predictor=pred, min_lookback=30)

        bars = _make_bars(60)
        result = pipeline.run(bars)
        for col in ["lower_1s", "upper_1s", "lower_2s", "upper_2s"]:
            assert col in result.columns


# ── KronosFactorCalculator ───────────────────────────────────────────


class TestKronosFactorCalculator:
    def test_specs_returns_three_factors(self):
        calc = KronosFactorCalculator()
        specs = calc.specs()
        assert len(specs) == 3
        names = {s.name for s in specs}
        assert "kronos_return_forecast" in names
        assert "kronos_range_forecast" in names
        assert "kronos_path_dispersion" in names

    def test_compute_returns_dataframe(self):
        cfg = KronosConfig(sample_count=5, pred_len=3, max_context=60)
        pred = _mock_predictor(5, 3)
        calc = KronosFactorCalculator(config=cfg, predictor=pred)

        bars = _make_bars(60)
        result = calc.compute(bars)

        assert isinstance(result, pd.DataFrame)
        assert "kronos_return_forecast" in result.columns
        assert "kronos_range_forecast" in result.columns
        assert "kronos_path_dispersion" in result.columns

    def test_early_rows_are_nan(self):
        cfg = KronosConfig(sample_count=3, pred_len=2, max_context=60)
        pred = _mock_predictor(3, 2)
        calc = KronosFactorCalculator(config=cfg, predictor=pred)

        bars = _make_bars(60)
        result = calc.compute(bars)
        assert result["kronos_return_forecast"].iloc[:30].isna().all()
        assert result["kronos_return_forecast"].iloc[30:].notna().any()


# ── KronosConfig ─────────────────────────────────────────────────────


class TestKronosConfig:
    def test_defaults(self):
        cfg = KronosConfig()
        assert cfg.model_name == "NeoQuasar/Kronos-mini"
        assert cfg.device == "cpu"
        assert cfg.sample_count == 10
        assert cfg.regime_confidence_weight + cfg.hmm_confidence_weight == pytest.approx(1.0)

    def test_from_yaml(self):
        cfg = KronosConfig.from_yaml()
        assert cfg.model_name == "NeoQuasar/Kronos-mini"
        assert cfg.pred_len == 5
