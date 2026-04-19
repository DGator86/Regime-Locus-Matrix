from __future__ import annotations

from rlm.training.regime_targets import derive_outcome_regime_label


def test_dominant_bull_maps_to_trend_up() -> None:
    label = derive_outcome_regime_label(
        {
            "bull_call_spread": 1.0,
            "bear_put_spread": -0.2,
            "iron_condor": 0.1,
            "calendar_spread": 0.0,
            "debit_spread": 0.2,
            "no_trade": 0.0,
        }
    )
    assert label == "trend_up_stable"


def test_dominant_calendar_maps_to_transition() -> None:
    label = derive_outcome_regime_label(
        {
            "bull_call_spread": 0.1,
            "bear_put_spread": 0.1,
            "iron_condor": 0.3,
            "calendar_spread": 0.7,
            "debit_spread": 0.2,
            "no_trade": 0.0,
        }
    )
    assert label == "transition"


def test_all_weak_real_strategies_forces_no_trade() -> None:
    label = derive_outcome_regime_label(
        {
            "bull_call_spread": -0.5,
            "bear_put_spread": -0.4,
            "iron_condor": -0.8,
            "calendar_spread": -0.3,
            "debit_spread": -0.6,
            "no_trade": 0.0,
        }
    )
    assert label == "no_trade"
