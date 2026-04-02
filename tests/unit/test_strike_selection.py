from rlm.roee.strategy_map import get_strategy_for_regime
from rlm.roee.strike_selection import build_legs_from_candidate


def test_bull_call_spread_strikes_are_ordered() -> None:
    candidate = get_strategy_for_regime(
        direction="bull",
        volatility="low_vol",
        liquidity="high_liquidity",
        dealer_flow="supportive",
    )
    legs = build_legs_from_candidate(
        candidate,
        current_price=5000.0,
        sigma=0.01,
        strike_increment=5.0,
    )

    assert len(legs) == 2
    assert legs[0].option_type == "call"
    assert legs[1].option_type == "call"
    assert legs[0].strike < legs[1].strike


def test_iron_condor_has_four_legs() -> None:
    candidate = get_strategy_for_regime(
        direction="range",
        volatility="low_vol",
        liquidity="high_liquidity",
        dealer_flow="supportive",
    )
    legs = build_legs_from_candidate(
        candidate,
        current_price=5000.0,
        sigma=0.01,
        strike_increment=5.0,
    )

    assert len(legs) == 4
