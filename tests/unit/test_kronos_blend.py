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
            row, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
        )
        assert result["confidence"] == pytest.approx(0.8)
        assert result["trade"] is True
        assert "kronos" not in result["model"]

    def test_kronos_only(self):
        row = _row(kronos_regime_agreement=0.7, kronos_transition_flag=False)
        result = compute_regime_modulators(
            row, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
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
            row, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
            hmm_confidence_weight=0.6, kronos_confidence_weight=0.4,
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
            confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        result_with_trans = compute_regime_modulators(
            row_with_trans, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        assert result_with_trans["confidence"] < result_no_trans["confidence"]

    def test_no_probs_no_kronos_blocks_trade(self):
        row = _row(foo=1)
        result = compute_regime_modulators(
            row, confidence_threshold=0.5, sizing_multiplier=1.0, transition_penalty=0.3,
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
            row, confidence_threshold=0.6, sizing_multiplier=1.0, transition_penalty=0.3,
            kronos_transition_penalty=0.3,
        )
        assert result["trade"] is False
