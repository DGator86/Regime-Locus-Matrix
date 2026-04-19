from __future__ import annotations

import pandas as pd

from rlm.training.strategy_targets_v2 import simulate_strategy_target_row_v2


def _row(**overrides):
    base = {
        "close": 100.0,
        "sigma": 0.02,
        "M_R_trans": 0.2,
        "M_trend_strength": 2.0,
        "M_D": 7.0,
        "M_L": 6.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _fwd(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def test_strategy_targets_v2_produces_all_columns_and_no_trade_zero() -> None:
    out = simulate_strategy_target_row_v2(_row(), _fwd([101, 102, 103]), strike_increment=5.0, horizon=3)
    assert set(out) == {
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
        "calendar_spread",
        "debit_spread",
        "no_trade",
    }
    assert out["no_trade"] == 0.0


def test_bullish_path_favors_bull_call_over_bear_put() -> None:
    out = simulate_strategy_target_row_v2(_row(), _fwd([102, 104, 107]), strike_increment=5.0, horizon=3)
    assert out["bull_call_spread"] > out["bear_put_spread"]


def test_bearish_path_favors_bear_put_over_bull_call() -> None:
    out = simulate_strategy_target_row_v2(_row(M_trend_strength=-2.0, M_D=3.0), _fwd([98, 96, 93]), strike_increment=5.0, horizon=3)
    assert out["bear_put_spread"] > out["bull_call_spread"]


def test_range_path_improves_condor() -> None:
    range_out = simulate_strategy_target_row_v2(_row(), _fwd([100.2, 99.8, 100.1]), strike_increment=5.0, horizon=3)
    trend_out = simulate_strategy_target_row_v2(_row(), _fwd([103, 106, 109]), strike_increment=5.0, horizon=3)
    assert range_out["iron_condor"] > trend_out["iron_condor"]


def test_high_transition_can_improve_calendar_vs_condor() -> None:
    out = simulate_strategy_target_row_v2(
        _row(M_R_trans=3.0, sigma=0.01),
        _fwd([100, 101, 99, 102, 100]),
        strike_increment=5.0,
        horizon=5,
    )
    assert out["calendar_spread"] > out["iron_condor"]
