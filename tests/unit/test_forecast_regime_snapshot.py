"""Regime transition JSON snapshot helper tests."""

from __future__ import annotations

import pandas as pd
import pytest

from rlm.regimes.forecast_regime_snapshot import (
    build_regime_transition_snapshot,
    regime_direction_equity,
    regime_transition_best_prob,
)


def test_regime_direction_equity_parses_pipe() -> None:
    assert regime_direction_equity("bull|a|b|c") == "bull"
    assert regime_direction_equity("") == ""


def test_regime_transition_best_prob_prefers_calibrated() -> None:
    snap = {
        "most_likely_next_prob": 0.2,
        "most_likely_next_prob_calibrated": 0.88,
    }
    assert regime_transition_best_prob(snap) == pytest.approx(0.88)


def test_build_hmm_snapshot() -> None:
    last = pd.Series(
        {
            "hmm_most_likely_next_prob": 0.35,
            "hmm_most_likely_next_prob_calibrated": 0.4,
            "hmm_most_likely_next_state": 2,
            "hmm_most_likely_next_label": "trend_up_stable",
            "hmm_expected_persistence": 0.7,
            "hmm_regime_transition_entropy": 0.5,
            "hmm_next_probs": [0.1, 0.2, 0.7],
            "hmm_state_label": "trend_up_stable",
        }
    )
    out = build_regime_transition_snapshot(last, live_model="hmm")
    assert out["family"] == "hmm"
    assert out["most_likely_next_prob"] == pytest.approx(0.35)
