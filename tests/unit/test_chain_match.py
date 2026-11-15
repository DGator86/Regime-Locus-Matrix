import pandas as pd

from rlm.data.option_chain import calculate_dte_from_expiry, normalize_option_chain
from rlm.roee.chain_match import match_legs_to_chain, select_chain_slice_for_decision
from rlm.roee.policy import select_trade, select_trade_from_strategy_name
from rlm.types.options import OptionLeg, TradeDecision


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


def _chain_with_deltas() -> pd.DataFrame:
    """Calls/puts where |delta|≈0.45 lives at 500, away from sigma spread strikes."""
    ts = pd.Timestamp("2026-06-03 10:00:00")
    expiry = pd.Timestamp("2026-06-03")
    rows = []
    for strike, call_delta in (
        (495, 0.20),
        (498, 0.30),
        (500, 0.45),
        (502, 0.55),
        (505, 0.70),
        (508, 0.80),
        (510, 0.85),
    ):
        for option_type, delta in (("call", call_delta), ("put", call_delta - 1.0)):
            rows.append(
                {
                    "timestamp": ts,
                    "underlying": "SPY",
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": strike,
                    "bid": 2.0,
                    "ask": 2.2,
                    "delta": delta,
                }
            )
    return pd.DataFrame(rows)


def test_short_dte_shared_delta_does_not_collapse_call_spread() -> None:
    """0DTE bull call spreads keep distinct sigma strikes under prefer_delta.

    ``_finalize_symbol`` stamps one ``target_delta`` on the decision whenever
    DTE max is <= 5. That target must not be applied to every leg: both calls
    would resolve to the 0.45-delta contract and the riskless-combo guard
    would skip the spread.
    """
    decision = select_trade(
        current_price=500.0,
        sigma=0.02,
        s_d=0.8,
        s_v=-0.5,
        s_l=0.7,
        s_g=0.8,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        strike_increment=1.0,
        short_dte=True,
    )
    assert decision.strategy_name == "0dte_bull_call_spread"
    assert [(leg.side, leg.option_type, leg.strike) for leg in decision.legs] == [
        ("long", "call", 502.0),
        ("short", "call", 508.0),
    ]
    decision.metadata = {**(decision.metadata or {}), "target_delta": 0.45}

    matched = match_legs_to_chain(
        decision=decision,
        chain_slice=_chain_with_deltas(),
        spot=500.0,
        prefer_delta=True,
    )
    assert matched.action == "enter", matched.rationale
    matched_legs = matched.metadata["matched_legs"]
    assert [(leg["side"], leg["option_type"], leg["strike"]) for leg in matched_legs] == [
        ("long", "call", 502.0),
        ("short", "call", 508.0),
    ]


def test_short_dte_shared_delta_keeps_straddle_at_intended_strikes() -> None:
    decision = select_trade(
        current_price=500.0,
        sigma=0.02,
        s_d=0.1,
        s_v=0.8,
        s_l=0.7,
        s_g=0.2,
        direction_regime="range",
        volatility_regime="high_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="destabilizing",
        regime_key="range|high_vol|high_liquidity|destabilizing",
        strike_increment=1.0,
        short_dte=True,
    )
    assert decision.strategy_name == "scalp_long_straddle"
    decision.metadata = {**(decision.metadata or {}), "target_delta": 0.35}

    matched = match_legs_to_chain(
        decision=decision,
        chain_slice=_chain_with_deltas(),
        spot=500.0,
        prefer_delta=True,
    )
    assert matched.action == "enter", matched.rationale
    assert [(leg["option_type"], leg["strike"]) for leg in matched.metadata["matched_legs"]] == [
        ("call", 500.0),
        ("put", 500.0),
    ]


def test_single_leg_still_matches_shared_target_delta() -> None:
    decision = TradeDecision(
        action="enter",
        strategy_name="long_call",
        legs=[OptionLeg(side="long", option_type="call", strike=510.0)],
        metadata={"target_delta": 0.45},
    )
    matched = match_legs_to_chain(
        decision=decision,
        chain_slice=_chain_with_deltas(),
        spot=500.0,
        prefer_delta=True,
    )
    assert matched.action == "enter", matched.rationale
    assert matched.metadata["matched_legs"][0]["strike"] == 500.0


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
