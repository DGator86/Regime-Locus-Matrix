import pandas as pd

from rlm.backtest.portfolio import Portfolio
from rlm.roee.chain_match import match_legs_to_chain
from rlm.roee.policy import select_trade


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


def test_portfolio_open_and_close_position() -> None:
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

    portfolio = Portfolio(initial_capital=100_000.0)
    position_id = portfolio.open_from_decision(
        timestamp=pd.Timestamp("2025-01-10 10:00:00"),
        underlying_symbol="SPY",
        underlying_price=5000.0,
        decision=matched,
        quantity=1,
    )

    assert position_id is not None
    assert len(portfolio.open_positions) == 1

    trade = portfolio.close_position(
        position_id=position_id,
        timestamp_exit=pd.Timestamp("2025-01-10 11:00:00"),
        underlying_price=5010.0,
        exit_reason="profit_target",
    )

    assert trade is not None
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_trades) == 1
