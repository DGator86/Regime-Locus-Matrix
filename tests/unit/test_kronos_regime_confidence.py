"""Tests for KronosRegimeConfidence using synthetic data (no model download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rlm.kronos.config import KronosConfig
from rlm.kronos.regime_confidence import (
    KronosRegimeConfidence,
    _classify_path,
    _direction_proxy,
    _volatility_proxy,
)


# ── Proxy helpers ────────────────────────────────────────────────────


def test_direction_proxy_positive():
    closes = np.array([101.0, 102.0, 103.0])
    proxy = _direction_proxy(100.0, closes)
    assert proxy > 0


def test_direction_proxy_negative():
    closes = np.array([99.0, 98.0, 97.0])
    proxy = _direction_proxy(100.0, closes)
    assert proxy < 0


def test_direction_proxy_zero_close():
    assert _direction_proxy(0.0, np.array([1.0])) == 0.0


def test_volatility_proxy():
    highs = np.array([102.0, 103.0])
    lows = np.array([98.0, 97.0])
    v = _volatility_proxy(100.0, highs, lows)
    assert isinstance(v, float)


def test_classify_path_returns_regime_key():
    path = np.array([
        [100, 102, 98, 101, 1000, 101000],
        [101, 103, 99, 102, 1100, 112200],
        [102, 104, 100, 103, 1200, 123600],
    ], dtype=float)
    rk = _classify_path(100.0, path)
    assert isinstance(rk, str)
    assert "|" in rk


# ── score_bar via mock predictor ─────────────────────────────────────


def _make_bars(n: int = 50) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": close - rng.rand(n) * 0.2,
        "high": close + rng.rand(n) * 0.5,
        "low": close - rng.rand(n) * 0.5,
        "close": close,
        "volume": rng.randint(100_000, 200_000, size=n),
    })


def _mock_predictor_factory(sample_count: int = 5, pred_len: int = 3):
    """Return a mock RLMKronosPredictor whose predict_paths returns stable arrays."""
    mock = MagicMock()

    def _predict_paths(df, future_timestamps=None):
        current_close = float(df["close"].iloc[-1])
        paths = np.empty((sample_count, pred_len, 6))
        for i in range(sample_count):
            drift = 0.005 * (i - sample_count // 2)
            for t in range(pred_len):
                c = current_close * (1 + drift * (t + 1))
                paths[i, t, :] = [c - 0.1, c + 0.3, c - 0.3, c, 150000, 150000 * c]
        return paths

    mock.predict_paths = _predict_paths
    return mock


def test_score_bar_produces_expected_keys():
    cfg = KronosConfig(sample_count=5, pred_len=3)
    mock_pred = _mock_predictor_factory(5, 3)
    krc = KronosRegimeConfidence(config=cfg, predictor=mock_pred)

    bars = _make_bars(50)
    result = krc.score_bar(bars, current_regime_key="bull|low_vol|low_liquidity|destabilizing")

    assert "kronos_confidence" in result
    assert "kronos_regime_agreement" in result
    assert "kronos_predicted_regime" in result
    assert "kronos_transition_flag" in result
    assert "kronos_forecast_return" in result
    assert "kronos_forecast_vol" in result

    assert 0.0 <= result["kronos_confidence"] <= 1.0
    assert 0.0 <= result["kronos_regime_agreement"] <= 1.0
    assert isinstance(result["kronos_transition_flag"], bool)


def test_score_bar_without_current_regime():
    cfg = KronosConfig(sample_count=5, pred_len=3)
    mock_pred = _mock_predictor_factory(5, 3)
    krc = KronosRegimeConfidence(config=cfg, predictor=mock_pred)

    bars = _make_bars(50)
    result = krc.score_bar(bars, current_regime_key=None)
    assert result["kronos_regime_agreement"] == result["kronos_confidence"]
    assert result["kronos_transition_flag"] is False


# ── annotate (batch) ─────────────────────────────────────────────────


def test_annotate_adds_columns():
    cfg = KronosConfig(sample_count=3, pred_len=2, max_context=50)
    mock_pred = _mock_predictor_factory(3, 2)
    krc = KronosRegimeConfidence(config=cfg, predictor=mock_pred)

    bars = _make_bars(60)
    bars["regime_key"] = "range|transition|low_liquidity|destabilizing"
    result = krc.annotate(bars, min_lookback=30)

    assert "kronos_confidence" in result.columns
    assert "kronos_regime_agreement" in result.columns
    assert result["kronos_confidence"].iloc[31:].notna().all()
    assert result["kronos_confidence"].iloc[:30].isna().all()
