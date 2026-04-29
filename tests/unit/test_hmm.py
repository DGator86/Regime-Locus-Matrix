import logging

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


def test_online_transition_update_returns_finite_row_stochastic_matrix() -> None:
    """online_transition_update must return a finite, non-negative, row-stochastic matrix."""
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=5,
            filter_backend="numpy",
        )
    ).fit(df, verbose=False)

    gamma = m.predict_proba_filtered(df)
    result = m.online_transition_update(gamma, step_size=0.1)

    assert result.shape == (4, 4)
    assert np.all(np.isfinite(result))
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(result >= 0.0)


def test_online_transition_update_mutates_model_transmat() -> None:
    """online_transition_update must mutate model.transmat_ and keep it row-stochastic."""
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=5,
            filter_backend="numpy",
        )
    ).fit(df, verbose=False)

    gamma = m.predict_proba_filtered(df)
    t_before = m.model.transmat_.copy()  # type: ignore[union-attr]
    m.online_transition_update(gamma, step_size=0.1)

    assert not np.allclose(m.model.transmat_, t_before, atol=1e-10)  # type: ignore[union-attr]
    assert np.allclose(m.model.transmat_.sum(axis=1), 1.0, atol=1e-5)  # type: ignore[union-attr]


def test_online_transition_update_preserves_permutation_alignment() -> None:
    """Permutation alignment: model.transmat_ is stored in raw-state space but calibrated_transmat returns permuted.

    After calling online_transition_update, permuted_transmat() should equal the updated values that
    were written back via the inverse permutation, and calibrated_transmat() should be consistent with
    that permuted view (up to pseudocount smoothing).
    """
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=7,
            filter_backend="numpy",
        )
    ).fit(df, verbose=False)

    assert m._state_permutation is not None, "Permutation must be set after fit"

    gamma = m.predict_proba_filtered(df)
    result = m.online_transition_update(gamma, step_size=0.05)

    # permuted_transmat reads from model.transmat_ through the permutation
    permuted = m.permuted_transmat()
    assert permuted.shape == (4, 4)
    assert np.allclose(permuted.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(permuted))

    # result must equal calibrated_transmat() (pseudocount applied on top of permuted view)
    calibrated = m.calibrated_transmat()
    assert calibrated.shape == (4, 4)
    assert np.allclose(calibrated, result, atol=1e-8)
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-5)

    # raw model.transmat_ is in un-permuted space; permuted_transmat must differ from it
    # when the permutation is non-trivial (not the identity)
    perm = m._state_permutation
    is_identity = all(old == new for old, new in perm.items())
    if not is_identity:
        assert not np.allclose(m.model.transmat_, permuted, atol=1e-8)  # type: ignore[union-attr]


def test_online_transition_update_zero_step_returns_calibrated_without_mutation() -> None:
    """step_size=0.0 must return calibrated_transmat without mutating model.transmat_."""
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=9,
            filter_backend="numpy",
        )
    ).fit(df, verbose=False)

    gamma = m.predict_proba_filtered(df)
    t_before = m.model.transmat_.copy()  # type: ignore[union-attr]
    result = m.online_transition_update(gamma, step_size=0.0)

    assert result.shape == (4, 4)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)
    # result must equal calibrated_transmat()
    assert np.allclose(result, m.calibrated_transmat(), atol=1e-8)
    # model.transmat_ must be unchanged
    assert np.allclose(m.model.transmat_, t_before)  # type: ignore[union-attr]


def test_online_transition_update_non_finite_step_raises() -> None:
    """Non-finite step_size must raise ValueError."""
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(n_states=4, n_iter=20, random_state=11, filter_backend="numpy")
    ).fit(df, verbose=False)
    gamma = m.predict_proba_filtered(df)
    with pytest.raises(ValueError, match="finite"):
        m.online_transition_update(gamma, step_size=float("nan"))


def test_online_transition_update_step_size_clamped_to_one() -> None:
    """step_size > 1.0 must be clamped to 1.0 (not raise, not extrapolate)."""
    df = _synthetic_scores(200)
    m = RLMHMM(
        HMMConfig(n_states=4, n_iter=20, random_state=13, filter_backend="numpy")
    ).fit(df, verbose=False)
    gamma = m.predict_proba_filtered(df)
    # Should not raise and result must be a valid stochastic matrix
    result = m.online_transition_update(gamma, step_size=5.0)
    assert result.shape == (4, 4)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(result >= 0.0)
