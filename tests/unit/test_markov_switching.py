import numpy as np
import pandas as pd

from rlm.forecasting.engines import HybridMarkovForecastPipeline
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching
from rlm.scoring.state_matrix import classify_state_matrix


def _synthetic_scores(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 5000 + np.cumsum(rng.normal(0.0, 2.0, size=n))
    df = pd.DataFrame(
        {
            "close": close,
            "S_D": np.sin(np.arange(n) / 11.0) + rng.normal(0, 0.15, size=n),
            "S_V": np.cos(np.arange(n) / 13.0) + rng.normal(0, 0.15, size=n),
            "S_L": np.sin(np.arange(n) / 17.0) + rng.normal(0, 0.15, size=n),
            "S_G": np.cos(np.arange(n) / 19.0) + rng.normal(0, 0.15, size=n),
        },
        index=idx,
    )
    return classify_state_matrix(df)


def test_markov_switching_fit_and_filter_shapes() -> None:
    df = _synthetic_scores()
    model = RLMMarkovSwitching(MarkovSwitchingConfig(n_states=3))
    model.fit(df.iloc[:180], verbose=False)
    probs = model.filter(df).to_numpy()

    assert probs.shape == (len(df), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert model.most_likely_state_filtered(df).shape == (len(df),)


def test_hybrid_markov_pipeline_adds_columns() -> None:
    df = _synthetic_scores()
    train_mask = pd.Series(df.index < df.index[180], index=df.index)
    out = HybridMarkovForecastPipeline(
        markov_config=MarkovSwitchingConfig(n_states=3),
    ).run(df, train_mask=train_mask)

    assert "markov_probs" in out.columns
    assert "markov_state" in out.columns
    assert "markov_state_label" in out.columns
    assert len(out["markov_probs"].iloc[-1]) == 3
