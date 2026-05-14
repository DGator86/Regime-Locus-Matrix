"""Tests for Kronos confidence blending in compute_regime_modulators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlm.roee.decision import compute_regime_modulators


def _row(**kwargs) -> pd.Series:
    """Build a ``pd.Series`` from keyword arguments (index = names, values = args)."""
    return pd.Series(kwargs)


class TestCompositeConfidence:
    def test_hmm_only(self):
        row = _row(hmm_probs=np.array([0.1, 0.8, 0.1]))
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
        )
        assert result["confidence"] == pytest.approx(0.8)
        assert result["trade"] is True
        assert "kronos" not in result["model"]

    def test_kronos_only(self):
        row = _row(kronos_regime_agreement=0.7, kronos_transition_flag=False)
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
        )
        assert result["confidence"] == pytest.approx(0.7)
        assert result["model"] == "kronos"
        assert result["trade"] is True

    def test_blended(self):
        row = _row(
            hmm_probs=np.array([0.2, 0.8]),
            kronos_regime_agreement=0.6,
            kronos_transition_flag=False,
        )
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
            hmm_confidence_weight=0.6,
            kronos_confidence_weight=0.4,
        )
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert result["confidence"] == pytest.approx(expected)
        assert "kronos" in result["model"]

    def test_transition_penalty_applied(self):
        row_no_trans = _row(
            hmm_probs=np.array([0.2, 0.8]),
            kronos_regime_agreement=0.6,
            kronos_transition_flag=False,
        )
        row_with_trans = _row(
            hmm_probs=np.array([0.2, 0.8]),
            kronos_regime_agreement=0.6,
            kronos_transition_flag=True,
        )
        result_no_trans = compute_regime_modulators(
            row_no_trans,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        result_with_trans = compute_regime_modulators(
            row_with_trans,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        assert result_with_trans["confidence"] < result_no_trans["confidence"]

    def test_no_probs_no_kronos_blocks_trade(self):
        row = _row(foo=1)
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.5,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
        )
        assert result["confidence"] == 0.0
        assert result["trade"] is False
        assert result["model"] == "none"

    def test_low_composite_blocks_trade(self):
        row = _row(
            hmm_probs=np.array([0.35, 0.3, 0.35]),
            kronos_regime_agreement=0.2,
            kronos_transition_flag=True,
        )
        result = compute_regime_modulators(
            row,
            confidence_threshold=0.6,
            sizing_multiplier=1.0,
            transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        assert result["trade"] is False


class TestTransitionAlertAndEnsemble:
    """Tests for the transition-alert and ensemble-confidence paths added to compute_regime_modulators."""

    def test_size_mult_decreases_as_hmm_alert_increases(self):
        """Higher hmm_transition_alert_probability must reduce size_mult."""
        base = dict(confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3)
        row_low = _row(hmm_probs=np.array([0.1, 0.8, 0.1]), hmm_transition_alert_probability=0.0)
        row_high = _row(hmm_probs=np.array([0.1, 0.8, 0.1]), hmm_transition_alert_probability=0.9)
        result_low = compute_regime_modulators(row_low, **base)
        result_high = compute_regime_modulators(row_high, **base)
        assert result_high["size_mult"] < result_low["size_mult"]

    def test_size_mult_decreases_as_markov_alert_increases(self):
        """Higher markov_transition_alert_probability must reduce size_mult."""
        base = dict(confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3)
        row_low = _row(hmm_probs=np.array([0.1, 0.8, 0.1]), markov_transition_alert_probability=0.0)
        row_high = _row(hmm_probs=np.array([0.1, 0.8, 0.1]), markov_transition_alert_probability=0.9)
        result_low = compute_regime_modulators(row_low, **base)
        result_high = compute_regime_modulators(row_high, **base)
        assert result_high["size_mult"] < result_low["size_mult"]

    def test_combined_hmm_and_markov_alerts_reduce_size_mult(self):
        """Both HMM and Markov alerts together must produce a lower size_mult than either alone."""
        base = dict(confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3)
        probs = np.array([0.1, 0.8, 0.1])
        row_none = _row(hmm_probs=probs)
        row_both = _row(hmm_probs=probs, hmm_transition_alert_probability=0.8, markov_transition_alert_probability=0.8)
        result_none = compute_regime_modulators(row_none, **base)
        result_both = compute_regime_modulators(row_both, **base)
        assert result_both["size_mult"] < result_none["size_mult"]

    def test_high_ensemble_confidence_increases_composite_vs_low(self):
        """regime_ensemble_confidence blends into composite: higher value must raise composite and size_mult."""
        base = dict(confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3)
        probs = np.array([0.2, 0.5, 0.3])  # hmm confidence = 0.5
        row_low_ens = _row(hmm_probs=probs, regime_ensemble_confidence=0.2)
        row_high_ens = _row(hmm_probs=probs, regime_ensemble_confidence=0.95)
        result_low = compute_regime_modulators(row_low_ens, **base)
        result_high = compute_regime_modulators(row_high_ens, **base)
        assert result_high["confidence"] > result_low["confidence"]
        assert result_high["size_mult"] > result_low["size_mult"]

    def test_alert_plus_ensemble_confidence_interaction(self):
        """A high transition alert should still reduce size_mult even when ensemble_confidence is elevated."""
        base = dict(confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3)
        probs = np.array([0.1, 0.8, 0.1])
        row_no_alert = _row(hmm_probs=probs, regime_ensemble_confidence=0.9)
        row_with_alert = _row(hmm_probs=probs, regime_ensemble_confidence=0.9, hmm_transition_alert_probability=0.9)
        result_no_alert = compute_regime_modulators(row_no_alert, **base)
        result_with_alert = compute_regime_modulators(row_with_alert, **base)
        assert result_with_alert["size_mult"] < result_no_alert["size_mult"]
