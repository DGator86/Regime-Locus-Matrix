"""Markov-switching transition matrix and one-step predictive columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.forecasting.engines import HybridMarkovForecastPipeline
from rlm.forecasting.hmm import RLMHMM
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching
from rlm.features.scoring.state_matrix import classify_state_matrix


def _synthetic_scores(n: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    obs = rng.normal(0, 0.4, size=(n, 4))
    df = pd.DataFrame(obs, columns=["S_D", "S_V", "S_L", "S_G"], index=idx)
    df["close"] = 100 + np.cumsum(rng.normal(0, 0.15, n))
    df["sigma"] = 0.01
    return classify_state_matrix(df)


def test_rlm_markov_transition_matrix_stochastic() -> None:
    df = _synthetic_scores(260)
    m = RLMMarkovSwitching(MarkovSwitchingConfig(n_states=3, transition_pseudocount=0.05))
    m.fit(df, verbose=False)
    t = m.transition_matrix()
    assert t.shape == (3, 3)
    assert np.allclose(t.sum(axis=1), 1.0, atol=1e-4)
    tc = m.calibrated_transition_matrix()
    gamma = m.filter(df).to_numpy(dtype=float)
    nxt = RLMHMM.one_step_predictive_probs(gamma, tc)
    assert nxt.shape == gamma.shape
    assert np.allclose(nxt.sum(axis=1), 1.0, atol=1e-4)


def test_hybrid_markov_pipeline_has_transition_columns() -> None:
    df = _synthetic_scores(240)
    train_mask = pd.Series(df.index < df.index[180], index=df.index)
    out = HybridMarkovForecastPipeline(
        markov_config=MarkovSwitchingConfig(n_states=3),
    ).run(df, train_mask=train_mask)
    assert "markov_next_probs" in out.columns
    assert "markov_most_likely_next_prob" in out.columns
    assert "markov_regime_transition_entropy" in out.columns


def test_hybrid_markov_pipeline_regime_ensemble_columns() -> None:
    """HybridMarkovForecastPipeline must produce regime_ensemble_probs/state/confidence."""
    df = _synthetic_scores(240)
    train_mask = pd.Series(df.index < df.index[180], index=df.index)
    out = HybridMarkovForecastPipeline(
        markov_config=MarkovSwitchingConfig(n_states=3),
    ).run(df, train_mask=train_mask)

    for col in ("regime_ensemble_probs", "regime_ensemble_state", "regime_ensemble_confidence"):
        assert col in out.columns, f"Missing column: {col}"

    ensemble = np.asarray(out["regime_ensemble_probs"].tolist(), dtype=float)
    assert ensemble.ndim == 2
    assert ensemble.shape[0] == len(out)
    assert np.all(np.isfinite(ensemble))
    assert np.allclose(ensemble.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(ensemble >= 0.0)

    conf = out["regime_ensemble_confidence"].to_numpy(dtype=float)
    assert np.all(np.isfinite(conf))
    assert np.all((conf >= 0.0) & (conf <= 1.0))
