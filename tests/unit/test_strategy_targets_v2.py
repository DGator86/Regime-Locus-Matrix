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
    out = simulate_strategy_target_row_v2(
        _row(), _fwd([101, 102, 103]), strike_increment=5.0, horizon=3
    )
    assert set(out) == {
        "bull_call_spread",
        "bear_put_spread",
        "iron_condor",
        "calendar_spread",
        "debit_spread",
        "no_trade",
    }
    assert out["no_trade"] == 0.0


def test_higher_illiquidity_worsens_targets() -> None:
    liquid = simulate_strategy_target_row_v2(
        _row(M_L=9.0),
        _fwd([102, 104, 107]),
        strike_increment=5.0,
        horizon=3,
    )
    illiquid = simulate_strategy_target_row_v2(
        _row(M_L=1.5),
        _fwd([102, 104, 107]),
        strike_increment=5.0,
        horizon=3,
    )
    assert illiquid["bull_call_spread"] < liquid["bull_call_spread"]


def test_path_exits_worsen_condor_on_breach_paths() -> None:
    with_exits = simulate_strategy_target_row_v2(
        _row(),
        _fwd([100, 106, 107, 108, 100]),
        strike_increment=5.0,
        horizon=5,
        use_path_exits=True,
    )
    without_exits = simulate_strategy_target_row_v2(
        _row(),
        _fwd([100, 106, 107, 108, 100]),
        strike_increment=5.0,
        horizon=5,
        use_path_exits=False,
    )
    assert with_exits["iron_condor"] < without_exits["iron_condor"]


def test_calendar_improves_with_higher_realized_vol_and_moderate_displacement() -> None:
    low_rv = simulate_strategy_target_row_v2(
        _row(sigma=0.03),
        _fwd([100.1, 100.0, 100.2, 100.1]),
        strike_increment=5.0,
        horizon=4,
    )
    high_rv = simulate_strategy_target_row_v2(
        _row(sigma=0.01),
        _fwd([100.0, 101.5, 99.0, 100.5]),
        strike_increment=5.0,
        horizon=4,
    )
    assert high_rv["calendar_spread"] > low_rv["calendar_spread"]


def test_debit_spread_target_flips_negative_on_conflicting_move() -> None:
    aligned = simulate_strategy_target_row_v2(
        _row(M_trend_strength=3.0, M_D=7.0),
        _fwd([101.0, 102.0, 103.0]),
        strike_increment=5.0,
        horizon=3,
    )
    conflict = simulate_strategy_target_row_v2(
        _row(M_trend_strength=3.0, M_D=7.0),
        _fwd([99.0, 98.0, 97.0]),
        strike_increment=5.0,
        horizon=3,
    )
    assert aligned["debit_spread"] > 0.0
    assert conflict["debit_spread"] < 0.0
