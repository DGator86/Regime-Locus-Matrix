from __future__ import annotations

import pandas as pd

from rlm.features.scoring.regime_model import RegimeModel
from rlm.roee.policy_models import PolicyConstraints, select_trade_from_models
from rlm.roee.strategy_value_model import StrategyValueModel


def test_bootstrap_regime_model_emits_probabilities() -> None:
    model = RegimeModel.with_bootstrap_coefficients()
    X = pd.DataFrame(
        [
            {
                "M_D": 8.0,
                "M_V": 4.0,
                "M_L": 7.0,
                "M_G": 8.0,
                "M_alignment": 9.0,
                "M_delta_neutral": 4.0,
                "M_R_trans": 1.0,
            }
        ]
    )
    probs = model.predict_proba(X)
    assert probs.shape == (1, 8)
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_value_model_and_policy_select_transition_calendar() -> None:
    regime_model = RegimeModel.with_bootstrap_coefficients()
    value_model = StrategyValueModel.with_bootstrap_coefficients()
    row = {
        "M_D": 5.0,
        "M_V": 6.0,
        "M_L": 6.0,
        "M_G": 5.0,
        "M_alignment": 0.0,
        "M_trend_strength": 0.0,
        "M_delta_neutral": 5.0,
        "M_R_trans": 7.0,
    }

    decision = select_trade_from_models(
        row,
        regime_model=regime_model,
        value_model=value_model,
        constraints=PolicyConstraints(trade_probability_threshold=0.1, min_edge=-10.0),
    )
    assert decision.trade
    assert decision.strategy == "calendar_spread"
