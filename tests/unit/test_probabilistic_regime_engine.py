"""Unit tests for ProbabilisticRegimeEngine and ProbabilisticRegimeEngineMTF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.forecasting.probabilistic_regime_engine import (
    PREConfig,
    ProbabilisticRegimeEngine,
    ProbabilisticRegimeEngineMTF,
    RegimeSignal,
    _bayesian_kronos_update,
    _build_htf_feature_lookup,
    _compute_attractiveness,
    _horizon_averaged_score,
    extract_pre_confidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ltf_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = np.array([[1.2, -0.9, 0.7, 0.6], [-1.0, 1.1, -0.5, -0.8], [0.2, 0.3, -1.1, 0.9]])
    labels = rng.integers(0, len(means), size=n)
    obs = means[labels] + rng.normal(0, 0.25, size=(n, 4))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame(obs, columns=["S_D", "S_V", "S_L", "S_G"], index=idx)
    df["close"] = 5000 + np.cumsum(rng.normal(0, 2, size=n))
    df["sigma"] = 0.01 + np.abs(rng.normal(0, 0.005, size=n))
    df["kronos_forecast"] = rng.normal(0.001, 0.005, size=n)
    df = classify_state_matrix(df)
    return df


def _make_htf_df(ltf_df: pd.DataFrame) -> pd.DataFrame:
    return ltf_df.resample("W-FRI").last().dropna(how="all")


def _small_config() -> PREConfig:
    from rlm.forecasting.hmm import HMMConfig
    return PREConfig(
        k_l=3,
        k_h=2,
        horizon=3,
        r_min=0.0,
        kronos_enabled=True,
        hmm_config=HMMConfig(n_states=3),
        htf_hmm_config=HMMConfig(n_states=2),
        min_state_samples=3,
        kronos_min_state_samples=2,
    )


# ---------------------------------------------------------------------------
# Pure mathematical helpers
# ---------------------------------------------------------------------------

class TestComputeAttractiveness:
    def test_normal_case(self):
        returns_by_state = {
            0: np.array([0.01, 0.02, 0.015, 0.012, 0.018]),   # positive mean
            1: np.array([-0.01, -0.02, -0.015, -0.012, -0.018]),  # negative mean
            2: np.array([]),  # no data → fallback 0.5
        }
        g = _compute_attractiveness(returns_by_state, k=3, r_min=0.0, min_samples=3)
        assert g.shape == (3,)
        assert g[0] > 0.5, "Positive-return state should have g > 0.5"
        assert g[1] < 0.5, "Negative-return state should have g < 0.5"
        assert g[2] == pytest.approx(0.5), "Empty state should default to 0.5"

    def test_output_bounded(self):
        rng = np.random.default_rng(0)
        returns_by_state = {i: rng.normal(0, 0.01, 20) for i in range(4)}
        g = _compute_attractiveness(returns_by_state, k=4, r_min=0.0, min_samples=5)
        assert np.all(g >= 0.0) and np.all(g <= 1.0)

    def test_min_samples_fallback(self):
        returns_by_state = {0: np.array([0.02, 0.03])}  # only 2 samples
        g = _compute_attractiveness(returns_by_state, k=1, r_min=0.0, min_samples=5)
        assert g[0] == pytest.approx(0.5)


class TestHorizonAveragedScore:
    def test_uniform_probs_uniform_transmat(self):
        K = 3
        g = np.array([0.8, 0.2, 0.5])
        T = np.ones((K, K)) / K
        p = np.ones(K) / K
        score, _ = _horizon_averaged_score(p, T, g, horizon=5)
        expected = float(np.mean(g))
        assert score == pytest.approx(expected, abs=1e-6)

    def test_score_in_unit_interval(self):
        rng = np.random.default_rng(7)
        K = 4
        T = rng.dirichlet(np.ones(K), size=K)
        g = rng.uniform(0, 1, K)
        p = rng.dirichlet(np.ones(K))
        score, path = _horizon_averaged_score(p, T, g, horizon=10)
        assert 0.0 <= score <= 1.0 + 1e-9
        assert path.shape == (10,)

    def test_horizon_1_equals_spot(self):
        K = 3
        g = np.array([0.9, 0.1, 0.5])
        T = np.eye(K)
        p = np.array([0.7, 0.2, 0.1])
        score, _ = _horizon_averaged_score(p, T, g, horizon=1)
        assert score == pytest.approx(float(p @ g), abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _horizon_averaged_score(np.ones(3), np.ones((3, 3)), np.ones(4), horizon=2)


class TestBayesianKronosUpdate:
    def test_consistent_state_raises_posterior(self):
        means = np.array([0.005, -0.005, 0.0])
        stds = np.array([0.002, 0.002, 0.002])
        prior = np.array([0.5, 0.3, 0.2])
        # A Kronos forecast very consistent with state 0
        posterior = _bayesian_kronos_update(prior, 0.005, means, stds)
        assert posterior[0] > prior[0], "State 0 posterior should increase"
        assert np.isclose(posterior.sum(), 1.0)

    def test_output_is_probability_vector(self):
        rng = np.random.default_rng(0)
        K = 5
        means = rng.normal(0, 0.005, K)
        stds = np.abs(rng.normal(0.003, 0.001, K)) + 1e-8
        prior = rng.dirichlet(np.ones(K))
        posterior = _bayesian_kronos_update(prior, 0.001, means, stds)
        assert np.isclose(posterior.sum(), 1.0)
        assert np.all(posterior >= 0.0)

    def test_degenerate_prior_survives(self):
        # Prior concentrated on one state; Kronos disagrees
        means = np.array([0.01, -0.01])
        stds = np.array([0.001, 0.001])
        prior = np.array([1.0, 0.0])
        # Should not crash even when one likelihood is near zero
        posterior = _bayesian_kronos_update(prior, 0.01, means, stds)
        assert np.isclose(posterior.sum(), 1.0)


# ---------------------------------------------------------------------------
# ProbabilisticRegimeEngine — single timeframe
# ---------------------------------------------------------------------------

class TestProbabilisticRegimeEngine:
    def test_fit_and_score_smoke(self, ltf_df):
        cfg = _small_config()
        engine = ProbabilisticRegimeEngine(cfg)
        engine.fit(ltf_df, train_timestamp="2024-01-01")
        assert engine.is_fitted
        assert engine.attractiveness is not None
        assert engine.attractiveness.shape == (3,)
        assert engine.transmat is not None
        assert engine.transmat.shape == (3, 3)

    def test_score_returns_valid_signal(self, ltf_df):
        cfg = _small_config()
        engine = ProbabilisticRegimeEngine(cfg)
        engine.fit(ltf_df)
        probs = engine._artefacts.hmm.predict_proba_filtered(ltf_df)
        kf = float(ltf_df["kronos_forecast"].iloc[-1])
        sig = engine.score(probs[-1], kronos_forecast=kf)
        assert isinstance(sig, RegimeSignal)
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.ltf_belief_raw.shape == (3,)
        assert sig.ltf_belief_post_kronos.shape == (3,)
        assert sig.expected_attractiveness_path.shape == (3,)
        assert 0.0 <= sig.instantaneous_attractiveness <= 1.0
        assert sig.current_most_likely_ltf_state in range(3)
        assert sig.htf_belief is None  # not MTF

    def test_score_without_kronos(self, ltf_df):
        engine = ProbabilisticRegimeEngine(_small_config())
        engine.fit(ltf_df)
        probs = engine._artefacts.hmm.predict_proba_filtered(ltf_df)
        sig = engine.score(probs[-1], kronos_forecast=None)
        assert np.allclose(sig.ltf_belief_raw, sig.ltf_belief_post_kronos)

    def test_score_without_fit_raises(self):
        engine = ProbabilisticRegimeEngine()
        with pytest.raises(RuntimeError):
            engine.score(np.array([0.5, 0.3, 0.2]))

    def test_confidence_bounded(self, ltf_df):
        engine = ProbabilisticRegimeEngine(_small_config())
        engine.fit(ltf_df)
        probs = engine._artefacts.hmm.predict_proba_filtered(ltf_df)
        for i in range(len(ltf_df)):
            sig = engine.score(probs[i])
            assert 0.0 <= sig.confidence <= 1.0

    def test_run_batch_adds_columns(self, ltf_df):
        engine = ProbabilisticRegimeEngine(_small_config())
        engine.fit(ltf_df)
        out = engine.run_batch(ltf_df)
        expected_cols = [
            "pre_confidence", "pre_spot_attractiveness", "pre_ltf_state",
            "pre_ltf_probs", "pre_ltf_probs_post_kronos", "pre_attractiveness_path",
        ]
        for col in expected_cols:
            assert col in out.columns, f"Missing column: {col}"
        assert (out["pre_confidence"] >= 0.0).all()
        assert (out["pre_confidence"] <= 1.0).all()

    def test_run_batch_without_fit_raises(self, ltf_df):
        with pytest.raises(RuntimeError):
            ProbabilisticRegimeEngine().run_batch(ltf_df)

    def test_attractiveness_changes_with_r_min(self, ltf_df):
        cfg_low = _small_config()
        cfg_low.r_min = -0.1
        cfg_high = _small_config()
        cfg_high.r_min = 0.05
        e_low = ProbabilisticRegimeEngine(cfg_low).fit(ltf_df)
        e_high = ProbabilisticRegimeEngine(cfg_high).fit(ltf_df)
        # Higher r_min → lower or equal attractiveness for each state
        assert np.all(e_high.attractiveness <= e_low.attractiveness + 1e-9)


# ---------------------------------------------------------------------------
# ProbabilisticRegimeEngineMTF
# ---------------------------------------------------------------------------

class TestProbabilisticRegimeEngineMTF:
    def test_fit_smoke(self, ltf_df, htf_df):
        cfg = _small_config()
        engine = ProbabilisticRegimeEngineMTF(cfg)
        engine.fit(ltf_df, htf_df)
        assert engine.is_fitted
        assert engine.attractiveness is not None
        assert engine.htf_attractiveness is not None

    def test_fit_without_htf_resamples_ltf(self, ltf_df):
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df=None)
        assert engine.is_fitted

    def test_build_htf_df_monthly_fallback_uses_supported_alias(self):
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        ltf_df = pd.DataFrame(
            {
                "S_D": [0.1, 0.2, 0.3],
                "S_V": [0.4, 0.5, 0.6],
                "S_L": [0.7, 0.8, 0.9],
                "S_G": [1.0, 1.1, 1.2],
            },
            index=idx,
        )
        engine = ProbabilisticRegimeEngineMTF(_small_config())

        htf = engine._build_htf_df(ltf_df)

        assert len(htf) == 1
        assert htf.index[0] == pd.Timestamp("2024-01-31")
        assert htf["S_D"].iloc[0] == pytest.approx(0.3)

    def test_update_returns_valid_signal(self, ltf_df, htf_df):
        cfg = _small_config()
        engine = ProbabilisticRegimeEngineMTF(cfg)
        engine.fit(ltf_df, htf_df)

        ltf_row = ltf_df[["S_D", "S_V", "S_L", "S_G"]].iloc[-1].values
        kf = float(ltf_df["kronos_forecast"].iloc[-1])
        sig = engine.update(ltf_row, kronos_forecast=kf, is_week_boundary=False)

        assert isinstance(sig, RegimeSignal)
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.htf_belief is not None
        assert sig.htf_belief.shape == (2,)
        assert sig.ltf_belief_raw.shape == (3,)
        assert sig.joint_belief is not None
        assert sig.joint_belief.shape == (2 * 3,)
        assert sig.current_most_likely_htf_state in range(2)

    def test_update_week_boundary(self, ltf_df, htf_df):
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df)
        ltf_row = ltf_df[["S_D", "S_V", "S_L", "S_G"]].iloc[-1].values
        htf_row = htf_df[["S_D", "S_V", "S_L", "S_G"]].iloc[-1].values
        sig = engine.update(ltf_row, is_week_boundary=True, new_htf_features=htf_row)
        assert 0.0 <= sig.confidence <= 1.0

    def test_update_week_boundary_no_htf_features(self, ltf_df, htf_df):
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df)
        ltf_row = ltf_df[["S_D", "S_V", "S_L", "S_G"]].iloc[-1].values
        sig = engine.update(ltf_row, is_week_boundary=True, new_htf_features=None)
        assert 0.0 <= sig.confidence <= 1.0

    def test_update_without_fit_raises(self):
        engine = ProbabilisticRegimeEngineMTF()
        with pytest.raises(RuntimeError):
            engine.update(np.zeros(4))

    def test_run_batch_adds_columns(self, ltf_df, htf_df):
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df)
        out = engine.run_batch(ltf_df, htf_df)
        expected_cols = [
            "pre_confidence", "pre_spot_attractiveness",
            "pre_ltf_state", "pre_htf_state",
            "pre_ltf_probs", "pre_ltf_probs_post_kronos",
            "pre_htf_probs", "pre_attractiveness_path",
        ]
        for col in expected_cols:
            assert col in out.columns, f"Missing column: {col}"
        assert (out["pre_confidence"] >= 0.0).all()
        assert (out["pre_confidence"] <= 1.0).all()

    def test_run_batch_same_length_as_input(self, ltf_df, htf_df):
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df)
        out = engine.run_batch(ltf_df, htf_df)
        assert len(out) == len(ltf_df)

    def test_htf_lookup_uses_score_columns_in_hmm_order(self):
        htf = pd.DataFrame(
            {
                "close": [5000.0],
                "sigma": [0.02],
                "S_D": [0.1],
                "S_V": [0.2],
                "S_L": [0.3],
                "S_G": [0.4],
            },
            index=pd.to_datetime(["2024-01-05"]),
        )
        lookup = _build_htf_feature_lookup(htf)
        values = next(iter(lookup.values()))
        assert values.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_run_batch_without_fit_raises(self, ltf_df):
        with pytest.raises(RuntimeError):
            ProbabilisticRegimeEngineMTF().run_batch(ltf_df)

    def test_confidence_varies_with_regime(self, ltf_df, htf_df):
        """Confidence should not be constant — it should respond to regime changes."""
        engine = ProbabilisticRegimeEngineMTF(_small_config())
        engine.fit(ltf_df, htf_df)
        out = engine.run_batch(ltf_df, htf_df)
        # There should be meaningful variation (std > 0)
        assert out["pre_confidence"].std() > 1e-6


# ---------------------------------------------------------------------------
# Decision layer integration
# ---------------------------------------------------------------------------

class TestExtractPreConfidence:
    def test_returns_float_when_present(self):
        row = pd.Series({"pre_confidence": 0.72})
        result = extract_pre_confidence(row)
        assert result == pytest.approx(0.72)

    def test_returns_none_when_absent(self):
        row = pd.Series({"hmm_probs": [0.5, 0.3, 0.2]})
        result = extract_pre_confidence(row)
        assert result is None

    def test_returns_none_for_nan(self):
        row = pd.Series({"pre_confidence": float("nan")})
        result = extract_pre_confidence(row)
        assert result is None

    def test_returns_none_for_inf(self):
        row = pd.Series({"pre_confidence": float("inf")})
        result = extract_pre_confidence(row)
        assert result is None


class TestComputeRegimeModulatorsWithPRE:
    """Verify that ``pre_confidence`` column triggers the PRE fast-path."""

    def test_pre_confidence_used_as_composite(self):
        from rlm.roee.decision import compute_regime_modulators

        row = pd.Series({
            "pre_confidence": 0.80,
            "hmm_probs": [0.3, 0.4, 0.3],
        })
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.2,
            use_pre_confidence=True,
        )
        assert result["model"] == "pre"
        assert result["confidence"] == pytest.approx(0.80)
        assert result["trade"] is True

    def test_pre_confidence_gates_when_below_threshold(self):
        from rlm.roee.decision import compute_regime_modulators

        row = pd.Series({"pre_confidence": 0.30})
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.2,
            use_pre_confidence=True,
        )
        assert result["trade"] is False
        assert result["model"] == "pre"

    def test_pre_confidence_disabled_falls_back_to_hmm(self):
        from rlm.roee.decision import compute_regime_modulators

        row = pd.Series({
            "pre_confidence": 0.80,
            "hmm_probs": [0.8, 0.1, 0.1],
        })
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.2,
            use_pre_confidence=False,
        )
        assert result["model"] != "pre"

    def test_size_mult_non_negative(self):
        from rlm.roee.decision import compute_regime_modulators

        row = pd.Series({"pre_confidence": 0.65})
        result = compute_regime_modulators(
            row, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.5
        )
        assert result["size_mult"] >= 0.0

    def test_epistemic_gate_overrides_pre(self):
        from rlm.roee.decision import compute_regime_modulators

        row = pd.Series({
            "pre_confidence": 0.90,
            "kronos_epistemic_uncertainty": 0.85,
        })
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.0,
            kronos_epistemic_disable_threshold=0.7,
            use_pre_confidence=True,
        )
        assert result["trade"] is False


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ltf_df():
    return _make_ltf_df(n=300)


@pytest.fixture(scope="module")
def htf_df(ltf_df):
    return _make_htf_df(ltf_df)
