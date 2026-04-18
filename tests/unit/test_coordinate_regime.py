from __future__ import annotations

import pandas as pd

from rlm.features.scoring.coordinate_mapper import add_market_coordinate_columns
from rlm.features.scoring.coordinate_regime import classify_regime_from_coordinates
from rlm.roee.decision import select_trade_for_row


def test_trend_up_regime() -> None:
    row = {
        "M_D": 8,
        "M_V": 4,
        "M_L": 7,
        "M_G": 8,
        "M_delta_neutral": 4,
        "M_R_trans": 1,
    }
    assert classify_regime_from_coordinates(row) == "trend_up_stable"


def test_transition_regime() -> None:
    row = {
        "M_D": 5,
        "M_V": 6,
        "M_L": 6,
        "M_G": 5,
        "M_delta_neutral": 5,
        "M_R_trans": 6,
    }
    assert classify_regime_from_coordinates(row) == "transition"


def test_add_market_coordinate_columns_adds_regime() -> None:
    df = pd.DataFrame(
        {
            "S_D": [0.6],
            "S_V": [-0.2],
            "S_L": [0.5],
            "S_G": [0.6],
        }
    )
    out = add_market_coordinate_columns(df)
    assert out["M_regime"].iloc[0] == "trend_up_stable"


def test_select_trade_for_row_skips_when_coordinate_router_no_trade() -> None:
    row = pd.Series(
        {
            "close": 5000.0,
            "sigma": 0.01,
            "S_D": 0.8,
            "S_V": -0.5,
            "S_L": 0.7,
            "S_G": 0.8,
            "direction_regime": "bull",
            "volatility_regime": "low_vol",
            "liquidity_regime": "high_liquidity",
            "dealer_flow_regime": "supportive",
            "regime_key": "bull|low_vol|high_liquidity|supportive",
            "M_D": 5.0,
            "M_V": 5.0,
            "M_L": 2.0,
            "M_G": 5.0,
            "M_delta_neutral": 0.0,
            "M_alignment": 0.0,
            "M_trend_strength": 0.0,
            "M_R_trans": 0.0,
            "M_regime": "no_trade",
        }
    )
    d = select_trade_for_row(row, strike_increment=5.0)
    assert d.action == "skip"
    assert d.strategy_name == "coordinate_router_no_trade"
