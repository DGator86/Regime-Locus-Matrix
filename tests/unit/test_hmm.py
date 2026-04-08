import numpy as np
import pandas as pd

from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.forecasting.pipeline import HybridForecastPipeline
from rlm.scoring.state_matrix import classify_state_matrix


def _synthetic_scores(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    means = np.array([
        [1.2, -0.9, 0.7, 0.6],
        [-1.0, 1.1, -0.5, -0.8],
        [0.2, 0.3, -1.1, 0.9],
    ])
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
    model = RLMHMM(
        HMMConfig(n_states=6, n_iter=25, random_state=11, filter_backend="numpy")
    ).fit(df, verbose=False)

    probs = model.predict_proba(df)
    states = model.most_likely_state(df)

    assert probs.shape == (250, 6)
    assert states.shape == (250,)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    filt = model.predict_proba_filtered(df)
    assert filt.shape == probs.shape
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-5)
    assert model.last_filter_backend == "numpy"


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