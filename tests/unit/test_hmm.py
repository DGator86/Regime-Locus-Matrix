import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from rlm.forecasting.engines import HybridForecastPipeline, _annotate_hmm_transition_fields, _annotate_regime_ensemble
from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.features.scoring.state_matrix import classify_state_matrix


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
    for name in ("filter_backend", "transition_pseudocount", "prefer_gpu", "online_em_step_size"):
        delattr(model.config, name)
    loaded = pickle.loads(pickle.dumps(model))

    filt = loaded.predict_proba_filtered(df)
    transmat = loaded.calibrated_transmat()
    updated_transmat = loaded.online_transition_update(filt)
    online_mats = loaded.causal_online_transition_matrices(filt)
    online = loaded.causal_online_transition_matrices(filt)

    assert loaded._state_permutation is None
    assert loaded.last_filter_backend in {"numpy", "numba"}
    assert loaded.config.filter_backend == "auto"
    assert loaded.config.transition_pseudocount == 0.1
    assert loaded.config.prefer_gpu is False
    assert loaded.config.online_em_step_size == 0.02
    assert filt.shape == (250, 6)
    assert online_mats.shape == (250, 6, 6)
    assert online.shape == (250, 6, 6)
    assert np.allclose(filt.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(transmat.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(updated_transmat.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(online_mats.sum(axis=2), 1.0, atol=1e-5)



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


def test_hmm_online_transition_annotations_are_causal_and_non_mutating() -> None:
    df = _synthetic_scores(220)

    m = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=0,
            filter_backend="numpy",
            transition_pseudocount=0.0,
            online_em_step_size=0.5,
        )
    )
    m.fit(df.iloc[:140], verbose=False)
    original = m.permuted_transmat().copy()
    gamma = m.predict_proba_filtered(df)

    mats = m.causal_online_transition_matrices(gamma)
    assert mats.shape == (len(df), 4, 4)
    assert np.allclose(m.permuted_transmat(), original)

    prefix_gamma = m.predict_proba_filtered(df.iloc[:170])
    prefix_mats = m.causal_online_transition_matrices(prefix_gamma)

    assert np.allclose(mats[:170], prefix_mats, atol=1e-8)


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


def test_online_transition_update_matches_final_prefix_path_matrix() -> None:
    """The batch update result must equal the last matrix in the prefix-causal path."""
    df = _synthetic_scores(200)
    m_path = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=5,
            filter_backend="numpy",
            online_em_step_size=0.1,
        )
    ).fit(df, verbose=False)
    m_batch = RLMHMM(
        HMMConfig(
            n_states=4,
            n_iter=20,
            random_state=5,
            filter_backend="numpy",
            online_em_step_size=0.1,
        )
    ).fit(df, verbose=False)

    gamma = m_path.predict_proba_filtered(df)
    path = m_path.online_transition_path(gamma)
    result = m_batch.online_transition_update(gamma)

    assert path.shape == (len(df), 4, 4)
    assert np.allclose(path[-1], result, atol=1e-8)
    assert np.allclose(path.sum(axis=2), 1.0, atol=1e-5)


def test_hmm_transition_annotations_do_not_leak_future_suffix() -> None:
    """Changing only future filtered probabilities must not rewrite prefix diagnostics."""
    n = 12
    prefix_len = 6
    idx = pd.RangeIndex(n)
    prefix = np.tile(np.array([[0.9, 0.1], [0.85, 0.15]]), (prefix_len // 2, 1))
    suffix_a = np.tile(np.array([[0.8, 0.2], [0.75, 0.25]]), ((n - prefix_len) // 2, 1))
    suffix_b = np.tile(np.array([[0.1, 0.9], [0.15, 0.85]]), ((n - prefix_len) // 2, 1))
    probs_a = np.vstack([prefix, suffix_a])
    probs_b = np.vstack([prefix, suffix_b])

    def _model() -> RLMHMM:
        hmm = RLMHMM(HMMConfig(n_states=2, transition_pseudocount=0.0, online_em_step_size=0.5))
        hmm.model = type(
            "DummyHMM",
            (),
            {"transmat_": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)},
        )()
        hmm.state_labels = ["bull_like", "bear_like"]
        return hmm

    df_a = pd.DataFrame({"hmm_state_label": ["bull_like"] * n}, index=idx)
    df_b = pd.DataFrame({"hmm_state_label": ["bull_like"] * n}, index=idx)

    _annotate_hmm_transition_fields(_model(), df_a, probs_a)
    _annotate_hmm_transition_fields(_model(), df_b, probs_b)

    for col in ("hmm_next_probs", "hmm_expected_persistence", "hmm_transition_alert_probability"):
        left = np.asarray(df_a[col].iloc[:prefix_len].tolist(), dtype=float)
        right = np.asarray(df_b[col].iloc[:prefix_len].tolist(), dtype=float)
        assert np.allclose(left, right, atol=1e-12), col


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


# ---------------------------------------------------------------------------
# _annotate_regime_ensemble tests
# ---------------------------------------------------------------------------


def _make_probs_df(
    n: int,
    n_states: int,
    *,
    with_close: bool = True,
    seed: int = 0,
    col: str = "hmm_probs",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    raw = rng.dirichlet(np.ones(n_states), size=n)
    df = pd.DataFrame(index=idx)
    df[col] = raw.tolist()
    if with_close:
        df["close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    return df


def test_annotate_regime_ensemble_single_source_columns_and_normalization() -> None:
    """Single hmm_probs source: all three columns are written, finite, non-negative, rows sum to 1."""
    df = _make_probs_df(100, 4, with_close=False)
    _annotate_regime_ensemble(df)

    for col in ("regime_ensemble_probs", "regime_ensemble_state", "regime_ensemble_confidence"):
        assert col in df.columns, f"Missing column: {col}"

    ensemble = np.asarray(df["regime_ensemble_probs"].tolist(), dtype=float)
    assert ensemble.shape == (100, 4)
    assert np.all(np.isfinite(ensemble))
    assert np.allclose(ensemble.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(ensemble >= 0.0)

    conf = df["regime_ensemble_confidence"].to_numpy(dtype=float)
    assert np.all(np.isfinite(conf))
    assert np.all((conf >= 0.0) & (conf <= 1.0))

    states = df["regime_ensemble_state"].to_numpy(dtype=int)
    assert states.shape == (100,)
    assert np.all((states >= 0) & (states < 4))


def test_annotate_regime_ensemble_no_sources_is_noop() -> None:
    """When neither hmm_probs nor markov_probs are present, no columns are added."""
    df = pd.DataFrame({"foo": [1, 2, 3]})
    _annotate_regime_ensemble(df)
    for col in ("regime_ensemble_probs", "regime_ensemble_state", "regime_ensemble_confidence"):
        assert col not in df.columns


def test_annotate_regime_ensemble_incompatible_shapes_raises() -> None:
    """hmm_probs(6 states) + markov_probs(3 states) must raise ValueError before stacking."""
    n = 50
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "hmm_probs": rng.dirichlet(np.ones(6), size=n).tolist(),
            "markov_probs": rng.dirichlet(np.ones(3), size=n).tolist(),
        }
    )
    with pytest.raises(ValueError, match="Incompatible"):
        _annotate_regime_ensemble(df)


def test_annotate_regime_ensemble_shock_reduces_confidence_vs_stable() -> None:
    """A large return shock at the last bar should temper ensemble confidence toward uniform."""
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    # Highly concentrated probabilities → high baseline confidence
    raw = np.zeros((n, 3))
    raw[:, 0] = 0.9
    raw[:, 1] = 0.05
    raw[:, 2] = 0.05

    # No-shock: completely flat price (all returns = 0, z-score = 0, cp_score = 0)
    df_stable = pd.DataFrame(
        {"hmm_probs": raw.tolist(), "close": np.full(n, 100.0)},
        index=idx,
    )
    _annotate_regime_ensemble(df_stable)

    # Shock: price is flat for all but the last bar, then jumps 100% → large return z-score at last row
    close_shock = np.concatenate([np.full(n - 1, 100.0), [200.0]])
    df_shock = pd.DataFrame({"hmm_probs": raw.tolist(), "close": close_shock}, index=idx)
    _annotate_regime_ensemble(df_shock)

    conf_stable = float(df_stable["regime_ensemble_confidence"].iloc[-1])
    conf_shock = float(df_shock["regime_ensemble_confidence"].iloc[-1])
    assert conf_shock < conf_stable, "Shock should temper ensemble confidence toward uniform"


def test_hmm_online_transition_annotations_are_prefix_stable() -> None:
    df = _synthetic_scores(260)
    train_mask = pd.Series(df.index < df.index[180], index=df.index)
    config = HMMConfig(n_states=4, n_iter=20, random_state=17, filter_backend="numpy", online_em_step_size=0.2)

    prefix = HybridForecastPipeline(hmm_config=config).run(df.iloc[:220], train_mask=train_mask.iloc[:220])
    extended = HybridForecastPipeline(hmm_config=config).run(df, train_mask=train_mask)

    prefix_next = np.asarray(prefix["hmm_next_probs"].tolist(), dtype=float)
    extended_next = np.asarray(extended.iloc[: len(prefix)]["hmm_next_probs"].tolist(), dtype=float)
    assert np.allclose(prefix_next, extended_next, atol=1e-10)
