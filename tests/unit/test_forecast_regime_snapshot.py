"""Regime transition JSON snapshot helper tests."""

from __future__ import annotations

import pandas as pd
import pytest

from rlm.regimes.forecast_regime_snapshot import (
    build_regime_transition_snapshot,
    label_aligned_transition_masses,
    position_directional_transition_mass,
    regime_direction_equity,
    regime_transition_best_prob,
    state_label_bull_alignment_weight,
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


def test_state_label_alignment_weights() -> None:
    assert state_label_bull_alignment_weight("trend_up_stable") == pytest.approx(1.0)
    assert state_label_bull_alignment_weight("trend_down_stable") == pytest.approx(0.0)


def test_label_aligned_masses_aggregate_full_row() -> None:
    probs = [0.2, 0.8]
    labs = ["trend_up_stable", "trend_down_stable"]
    b, br = label_aligned_transition_masses(probs, labs)
    assert b == pytest.approx(0.2)
    assert br == pytest.approx(0.8)


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
            "hmm_state_labels": ["trend_up_stable", "trend_down_stable", "range_compression"],
            "hmm_state_label": "trend_up_stable",
        }
    )
    out = build_regime_transition_snapshot(last, live_model="hmm")
    assert out["family"] == "hmm"
    assert out["most_likely_next_prob"] == pytest.approx(0.35)
    assert out["state_labels"] == ["trend_up_stable", "trend_down_stable", "range_compression"]
    bm, br = out["next_label_aligned_bull_mass"], out["next_label_aligned_bear_mass"]
    assert bm is not None and br is not None
    assert bm == pytest.approx(0.45)
    assert position_directional_transition_mass(out, "bull") == pytest.approx(0.45)
    assert position_directional_transition_mass(out, "bear") == pytest.approx(0.55)
