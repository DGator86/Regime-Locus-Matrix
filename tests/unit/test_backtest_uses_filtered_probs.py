"""Regression guard: HybridForecastPipeline must use predict_proba_filtered, not predict_proba.

predict_proba uses the Viterbi/smoothed (two-pass) probabilities, which incorporate future
observations and would constitute look-ahead bias in backtest and live paths. The forward-only
filter (predict_proba_filtered) is the only causal option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.forecasting.engines import HybridForecastPipeline
from rlm.forecasting.hmm import RLMHMM, HMMConfig


def _synthetic_scores(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    means = np.array([[1.2, -0.9, 0.7, 0.6], [-1.0, 1.1, -0.5, -0.8], [0.2, 0.3, -1.1, 0.9]])
    labels = rng.integers(0, len(means), size=n)
    obs = means[labels] + rng.normal(0, 0.25, size=(n, 4))
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame(obs, columns=["S_D", "S_V", "S_L", "S_G"], index=idx)
    df["close"] = 5000 + np.cumsum(rng.normal(0, 2, size=n))
    df["sigma"] = 0.01 + np.abs(rng.normal(0, 0.005, size=n))
    return classify_state_matrix(df)


def test_hybrid_pipeline_uses_filtered_not_smoothed_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """HybridForecastPipeline must call predict_proba_filtered, never the smoothing predict_proba."""
    smoothed_calls: list[str] = []

    original_predict_proba = RLMHMM.predict_proba

    def spy_predict_proba(self: RLMHMM, df: pd.DataFrame) -> np.ndarray:
        smoothed_calls.append("predict_proba called")
        return original_predict_proba(self, df)

    monkeypatch.setattr(RLMHMM, "predict_proba", spy_predict_proba)

    filtered_calls: list[str] = []
    original_filtered = RLMHMM.predict_proba_filtered

    def spy_predict_proba_filtered(self: RLMHMM, df: pd.DataFrame) -> np.ndarray:
        filtered_calls.append("predict_proba_filtered called")
        return original_filtered(self, df)

    monkeypatch.setattr(RLMHMM, "predict_proba_filtered", spy_predict_proba_filtered)

    df = _synthetic_scores(220)
    train_mask = pd.Series(df.index < df.index[160], index=df.index)
    HybridForecastPipeline(
        hmm_config=HMMConfig(n_states=4, n_iter=15, random_state=7, filter_backend="numpy"),
    ).run(df, train_mask=train_mask)

    assert filtered_calls, "predict_proba_filtered was never called — look-ahead bias risk"
    assert not smoothed_calls, (
        "predict_proba (smoothed/two-pass) was called from backtest/live path — "
        "this introduces look-ahead bias. Use predict_proba_filtered instead."
    )
