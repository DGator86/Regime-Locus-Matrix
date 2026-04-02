import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.backtest.lifecycle import LifecycleConfig, days_to_expiry, is_at_or_past_expiry, should_force_close_before_expiry
from rlm.backtest.portfolio import Portfolio
from rlm.types.options import TradeDecision


def _decision_with_one_leg(expiry: str = "2025-01-12") -> TradeDecision:
    return TradeDecision(
        action="enter",
        strategy_name="test",
        regime_key="test_regime",
        target_profit_pct=10.0,
        max_risk_pct=0.05,
        metadata={
            "matched_legs": [
                {
                    "side": "long",
                    "option_type": "call",
                    "strike": 100.0,
                    "expiry": expiry,
                    "bid": 1.0,
                    "ask": 1.2,
                    "mid": 1.1,
                }
            ]
        },
    )


def test_lifecycle_dte_helpers() -> None:
    cfg = LifecycleConfig(force_close_dte=1)
    assert days_to_expiry(timestamp=pd.Timestamp("2025-01-10"), expiry="2025-01-12") == 2
    assert should_force_close_before_expiry(
        timestamp=pd.Timestamp("2025-01-11"), expiry="2025-01-12", config=cfg
    )
    assert is_at_or_past_expiry(timestamp=pd.Timestamp("2025-01-12"), expiry="2025-01-12")


def test_portfolio_applies_commissions_on_open_and_close() -> None:
    portfolio = Portfolio(
        initial_capital=10_000.0,
        lifecycle_config=LifecycleConfig(commission_per_contract=1.0),
    )
    decision = _decision_with_one_leg()

    position_id = portfolio.open_from_decision(
        timestamp=pd.Timestamp("2025-01-10 10:00:00"),
        underlying_symbol="SPY",
        underlying_price=100.0,
        decision=decision,
        quantity=1,
    )
    assert position_id is not None

    pos = portfolio.open_positions[position_id]
    pos.current_exit_value_cache = pos.entry_cost + 10.0

    trade = portfolio.close_position(
        position_id=position_id,
        timestamp_exit=pd.Timestamp("2025-01-10 11:00:00"),
        underlying_price=101.0,
        exit_reason="test",
    )
    assert trade is not None
    # pnl is reduced by the close commission that is netted from exit value
    assert trade.pnl < 10.0


def test_engine_forces_close_before_expiry(monkeypatch) -> None:
    timestamps = pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"])
    features = pd.DataFrame(
        {
            "close": [100.0, 100.5, 101.0],
            "sigma": [0.01, 0.01, 0.01],
            "S_D": [0.8, 0.8, 0.8],
            "S_V": [-0.3, -0.3, -0.3],
            "S_L": [0.7, 0.7, 0.7],
            "S_G": [0.8, 0.8, 0.8],
            "direction_regime": ["bull", "bull", "bull"],
            "volatility_regime": ["low_vol", "low_vol", "low_vol"],
            "liquidity_regime": ["high_liquidity", "high_liquidity", "high_liquidity"],
            "dealer_flow_regime": ["supportive", "supportive", "supportive"],
            "regime_key": ["r", "r", "r"],
            "lower_1s": [95.0, 95.0, 95.0],
            "upper_1s": [105.0, 105.0, 105.0],
        },
        index=timestamps,
    )

    chain_rows = []
    for ts in timestamps:
        chain_rows.append(
            {
                "timestamp": ts,
                "underlying": "SPY",
                "expiry": pd.Timestamp("2025-01-12"),
                "option_type": "call",
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.2,
            }
        )
    chain = pd.DataFrame(chain_rows)

    monkeypatch.setattr("rlm.backtest.engine.select_trade", lambda **_: _decision_with_one_leg())
    monkeypatch.setattr("rlm.backtest.engine.match_legs_to_chain", lambda decision, chain_slice: decision)
    monkeypatch.setattr("rlm.backtest.engine.select_nearest_expiry_slice", lambda row_chain, dte_min, dte_max: row_chain)

    engine = BacktestEngine(
        lifecycle_config=LifecycleConfig(
            force_close_dte=1,
            close_at_expiry_if_open=True,
            max_holding_bars=None,
            one_trade_per_bar=True,
        )
    )

    _, trades, _ = engine.run(features, chain)

    assert not trades.empty
    assert "forced_pre_expiry" in set(trades["exit_reason"])
