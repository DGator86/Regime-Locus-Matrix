from __future__ import annotations

import pandas as pd

from rlm.features.scoring.regime_model import RegimeModel
from rlm.features.scoring.regime_model_temporal import TemporalRegimeModel
from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.roee.strategy_value_model_temporal import TemporalStrategyValueModel


def _synthetic_coordinate_frame(n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "M_D": 5.0 + (i % 4),
                "M_V": 4.8 + (i % 3) * 0.4,
                "M_L": 5.1 + (i % 2) * 0.3,
                "M_G": 5.0 + (i % 5) * 0.2,
                "M_trend_strength": float((i % 5) - 2),
                "M_dealer_control": float(i % 4),
                "M_alignment": float((i % 3) - 1),
                "M_delta_neutral": float(i % 7),
                "M_R_trans": float(i % 3),
            }
        )
    return pd.DataFrame(rows)


def test_temporal_regime_model_predicts_same_length_as_input() -> None:
    X = _synthetic_coordinate_frame(20)
    base = RegimeModel.with_bootstrap_coefficients()
    model = TemporalRegimeModel(base_model=base, window=5, smoothing_alpha=0.25)
    probs = model.predict_proba(X)
    assert probs.shape[0] == len(X)


def test_temporal_value_model_outputs_expected_shape() -> None:
    X = _synthetic_coordinate_frame(20)
    base = StrategyValueModel.with_bootstrap_coefficients()
    model = TemporalStrategyValueModel(base_model=base, window=5)
    pred = model.predict_expected_values(X)
    assert pred.shape[0] == len(X)
