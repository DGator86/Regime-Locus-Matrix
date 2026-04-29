import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from rlm.forecasting.engines import HybridForecastPipeline
from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.scoring.state_matrix import classify_state_matrix


def _synthetic_scores(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    means = np.array(
        [
            [1.2, -0.9, 0.7, 0.6],
            [-1.0, 1.1, -0.5, -0.8],
            [0.2, 0.3, -1.1, 0.9],
        ]
    )
    labels = rng.integers(0, len(means), size=n)
    obs = means[labels] + rng.normal(0, 0.25, size=(n, 4))

    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame(obs, columns=["S_D", "S_V", "S_L", "S_G"], index=idx)
    df["close"] = 5000 + np.cumsum(rng.normal(0, 2, size=n))
    df["sigma"] = 0.01 + np.abs(rng.normal(0, 0.005, size=n))
    df = classify_state_matrix(df)
    return df


def test_rlm_hmm_fit_and_predict_shape() -> None:
    df = _synthetic_scores(250)
    model = RLMHMM(HMMConfig(n_states=6, n_iter=25, random_state=11, filter_backend="numpy")).fit(df, verbose=False)

    probs = model.predict_proba(df)
    states = model.most_likely_state(df)

    assert probs.shape == (250, 6)
    assert states.shape == (250,)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    filt = model.predict_proba_filtered(df)
    assert filt.shape == probs.shape
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-5)
    assert model.last_filter_backend == "numpy"


def test_rlm_hmm_legacy_pickle_without_state_permutation_still_predicts() -> None:
    df = _synthetic_scores(250)
    model = RLMHMM(HMMConfig(n_states=6, n_iter=25, random_state=11, filter_backend="numpy")).fit(df, verbose=False)
    delattr(model, "_state_permutation")
    loaded = pickle.loads(pickle.dumps(model))

    probs = loaded.predict_proba(df)
    states = loaded.most_likely_state(df)
    filt = loaded.predict_proba_filtered(df)

    assert loaded._state_permutation is None
    assert probs.shape == (250, 6)
    assert states.shape == (250,)
    assert filt.shape == probs.shape
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-5)


def test_rlm_hmm_legacy_pickle_without_new_config_fields_still_predicts() -> None:
    df = _synthetic_scores(250)
    model = RLMHMM(HMMConfig(n_states=6, n_iter=25, random_state=11, filter_backend="numpy")).fit(df, verbose=False)
    for name in ("_state_permutation", "last_filter_backend"):
        delattr(model, name)
    for name in ("filter_backend", "transition_pseudocount", "prefer_gpu"):
        delattr(model.config, name)
    loaded = pickle.loads(pickle.dumps(model))

    filt = loaded.predict_proba_filtered(df)
    transmat = loaded.calibrated_transmat()

    assert loaded._state_permutation is None
    assert loaded.last_filter_backend in {"numpy", "numba"}
    assert loaded.config.filter_backend == "auto"
    assert loaded.config.transition_pseudocount == 0.1
    assert loaded.config.prefer_gpu is False
    assert filt.shape == (250, 6)
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(transmat.sum(axis=1), 1.0, atol=1e-5)



def test_hybrid_forecast_pipeline_adds_hmm_columns() -> None:
    df = _synthetic_scores(220)
    train_mask = pd.Series(df.index < df.index[160], index=df.index)

    out = HybridForecastPipeline(
        hmm_config=HMMConfig(n_states=6, n_iter=15, random_state=3),
    ).run(df, train_mask=train_mask)

    assert "hmm_probs" in out.columns
    assert "hmm_state" in out.columns
    assert "hmm_state_label" in out.columns
    assert len(out["hmm_probs"].iloc[-1]) == 6
    assert "hmm_next_probs" in out.columns
    assert "hmm_regime_transition_entropy" in out.columns
    assert "hmm_expected_persistence" in out.columns
    assert "hmm_most_likely_next_state" in out.columns
    assert "hmm_most_likely_next_prob" in out.columns


def test_rlm_hmm_fit_suppresses_hmmlearn_nonmonotone_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """hmmlearn logs WARNING when a single EM step dips; we silence it during fit (journal noise)."""
    caplog.set_level(logging.WARNING, logger="hmmlearn.base")
    df = _synthetic_scores(400)
    RLMHMM(HMMConfig(n_states=6, n_iter=40, random_state=99, filter_backend="numpy")).fit(df, verbose=False)
    noisy = [r for r in caplog.records if "Model is not converging" in r.getMessage()]
    assert not noisy


def test_hmm_calibrated_transmat_and_one_step_predictive() -> None:
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=0,
            filter_backend="numpy",
            transition_pseudocount=0.05,
        )
    )
    m.fit(df, verbose=False)
    t = m.calibrated_transmat()
    assert t.shape == (4, 4)
    assert np.allclose(t.sum(axis=1), 1.0, atol=1e-5)
    gamma = m.predict_proba_filtered(df)
    nxt = RLMHMM.one_step_predictive_probs(gamma, t)
    assert nxt.shape == gamma.shape
    assert np.allclose(nxt.sum(axis=1), 1.0, atol=1e-5)
