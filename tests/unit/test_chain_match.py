import pandas as pd

from rlm.data.option_chain import calculate_dte_from_expiry, normalize_option_chain
from rlm.roee.chain_match import match_legs_to_chain, select_chain_slice_for_decision
from rlm.roee.policy import select_trade, select_trade_from_strategy_name


def make_chain() -> pd.DataFrame:
    ts = pd.Timestamp("2025-01-10 10:00:00")
    expiry = pd.Timestamp("2025-02-14")

    rows = []
    for option_type in ["call", "put"]:
        for strike in [4950, 4975, 5000, 5025, 5050, 5075]:
            rows.append(
                {
                    "timestamp": ts,
                    "underlying": "SPY",
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike,
                    "bid": 8.0 if option_type == "call" else 7.5,
                    "ask": 8.4 if option_type == "call" else 7.9,
                }
            )
    return pd.DataFrame(rows)


def test_match_legs_to_chain_for_bull_spread() -> None:
    decision = select_trade(
        current_price=5000.0,
        sigma=0.01,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=25.0,
    )

    matched = match_legs_to_chain(decision=decision, chain_slice=make_chain())
    assert matched.action == "enter"
    assert "matched_legs" in matched.metadata
    assert len(matched.metadata["matched_legs"]) == 2


def test_calendar_spread_matches_near_and_far_expiries() -> None:
    ts = pd.Timestamp("2025-01-10 10:00:00")
    rows = []
    for expiry in [pd.Timestamp("2025-02-07"), pd.Timestamp("2025-02-21")]:
        for strike in [4975, 5000, 5025]:
            rows.append(
                {
                    "timestamp": ts,
                    "underlying": "SPY",
                    "expiry": expiry,
                    "option_type": "call",
                    "strike": strike,
                    "bid": 8.0,
                    "ask": 8.4,
                }
            )
    chain = normalize_option_chain(pd.DataFrame(rows))

    decision = select_trade_from_strategy_name(
        strategy_name="calendar_spread",
        current_price=5000.0,
        sigma=0.01,
        s_d=0.2,
        s_v=-0.2,
        s_l=0.6,
        s_g=0.5,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=25.0,
    )

    assert [(leg.side, leg.strike, leg.expiry) for leg in decision.legs] == [
        ("short", 5000.0, "near"),
        ("long", 5000.0, "far"),
    ]

    chain_slice = select_chain_slice_for_decision(chain, decision)
    assert chain_slice["expiry"].nunique() == 2

    matched = match_legs_to_chain(decision=decision, chain_slice=chain_slice)
    assert matched.action == "enter"
    matched_legs = matched.metadata["matched_legs"]
    assert [leg["side"] for leg in matched_legs] == ["short", "long"]
    assert {leg["strike"] for leg in matched_legs} == {5000.0}
    assert matched_legs[0]["expiry"] == "2025-02-07"
    assert matched_legs[1]["expiry"] == "2025-02-21"


def test_calculate_dte_same_day() -> None:
    assert calculate_dte_from_expiry("2025-01-10", "2025-01-10") == 0.0


def test_calculate_dte_thirty_days() -> None:
    assert calculate_dte_from_expiry("2025-02-09", "2025-01-10") == 30.0


def test_calculate_dte_intraday_timestamps_normalize() -> None:
    # Time component should not affect calendar-day DTE.
    assert calculate_dte_from_expiry("2025-01-11 16:00", "2025-01-10 09:59") == 1.0


def test_calculate_dte_at_time_stop_threshold() -> None:
    # force_exit_dte default is 2 — verify boundary value.
    assert calculate_dte_from_expiry("2025-01-12", "2025-01-10") == 2.0
