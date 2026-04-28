"""Integration tests for expiry settlement in the portfolio and engine."""

from __future__ import annotations

import pandas as pd
import pytest

from rlm.backtest.commission import CommissionConfig, CommissionModel
from rlm.backtest.engine import BacktestEngine
from rlm.backtest.lifecycle import ExpiryLiquidationPolicy, LifecycleConfig
from rlm.backtest.portfolio import Portfolio
from rlm.types.options import TradeDecision


def _single_leg_decision(
    expiry: str = "2025-01-12",
    side: str = "long",
    option_type: str = "call",
    strike: float = 100.0,
    bid: float = 1.0,
    ask: float = 1.2,
) -> TradeDecision:
    return TradeDecision(
        action="enter",
        strategy_name="test",
        regime_key="test_regime",
        target_profit_pct=10.0,
        max_risk_pct=0.05,
        metadata={
            "matched_legs": [
                {
                    "side": side,
                    "option_type": option_type,
                    "strike": strike,
                    "expiry": expiry,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# Portfolio.expiry_settle_position
# ---------------------------------------------------------------------------


class TestExpirySettlePosition:
    def _make_portfolio(self) -> Portfolio:
        return Portfolio(
            initial_capital=10_000.0,
            contract_multiplier=100,
            lifecycle_config=LifecycleConfig(commission_per_contract=0.0),
        )

    def test_long_call_itm_settlement_increases_cash(self) -> None:
        portfolio = self._make_portfolio()
        decision = _single_leg_decision(
            strike=95.0, option_type="call", side="long", bid=5.0, ask=5.4
        )
        position_id = portfolio.open_from_decision(
            timestamp=pd.Timestamp("2025-01-10"),
            underlying_symbol="SPY",
            underlying_price=100.0,
            decision=decision,
            quantity=1,
        )
        assert position_id is not None
        cash_before = portfolio.cash

        trade = portfolio.expiry_settle_position(
            position_id=position_id,
            timestamp_exit=pd.Timestamp("2025-01-12"),
            underlying_price=105.0,  # 10 pts ITM
        )

        assert trade is not None
        assert trade.exit_reason == "expiry_settlement"
        # Long call intrinsic = 10; cash_impact = 10 * 100 = 1000
        assert portfolio.cash == pytest.approx(cash_before + 1000.0)
        assert position_id not in portfolio.open_positions

    def test_long_call_otm_expires_worthless(self) -> None:
        portfolio = self._make_portfolio()
        decision = _single_leg_decision(
            strike=110.0, option_type="call", side="long", bid=0.5, ask=0.7
        )
        position_id = portfolio.open_from_decision(
            timestamp=pd.Timestamp("2025-01-10"),
            underlying_symbol="SPY",
            underlying_price=100.0,
            decision=decision,
            quantity=1,
        )
        assert position_id is not None
        cash_before = portfolio.cash

        trade = portfolio.expiry_settle_position(
            position_id=position_id,
            timestamp_exit=pd.Timestamp("2025-01-12"),
            underlying_price=105.0,  # OTM
        )

        assert trade is not None
        assert trade.exit_reason == "expiry_settlement"
        assert trade.exit_value == pytest.approx(0.0)
        # No cash impact for OTM
        assert portfolio.cash == pytest.approx(cash_before)

    def test_short_call_itm_assignment_debits_cash(self) -> None:
        portfolio = self._make_portfolio()
        # Short call: portfolio receives premium on entry (negative entry cost)
        decision = _single_leg_decision(
            strike=100.0, option_type="call", side="short", bid=3.0, ask=3.4
        )
        position_id = portfolio.open_from_decision(
            timestamp=pd.Timestamp("2025-01-10"),
            underlying_symbol="SPY",
            underlying_price=100.0,
            decision=decision,
            quantity=1,
        )
        assert position_id is not None
        cash_before = portfolio.cash

        trade = portfolio.expiry_settle_position(
            position_id=position_id,
            timestamp_exit=pd.Timestamp("2025-01-12"),
            underlying_price=108.0,  # 8 pts ITM
        )

        assert trade is not None
        assert trade.metadata.get("assignment_occurred") is True
        # Assignment: portfolio owes 8 * 100 = 800
        assert portfolio.cash == pytest.approx(cash_before - 800.0)

    def test_short_call_otm_full_profit(self) -> None:
        portfolio = self._make_portfolio()
        decision = _single_leg_decision(
            strike=110.0, option_type="call", side="short", bid=1.0, ask=1.4
        )
        position_id = portfolio.open_from_decision(
            timestamp=pd.Timestamp("2025-01-10"),
            underlying_symbol="SPY",
            underlying_price=100.0,
            decision=decision,
            quantity=1,
        )
        assert position_id is not None
        cash_before = portfolio.cash

        trade = portfolio.expiry_settle_position(
            position_id=position_id,
            timestamp_exit=pd.Timestamp("2025-01-12"),
            underlying_price=105.0,  # OTM
        )

        assert trade is not None
        assert trade.metadata.get("assignment_occurred") is False
        # No assignment, no cash impact
        assert portfolio.cash == pytest.approx(cash_before)

    def test_unknown_position_id_returns_none(self) -> None:
        portfolio = self._make_portfolio()
        result = portfolio.expiry_settle_position(
            position_id="nonexistent",
            timestamp_exit=pd.Timestamp("2025-01-12"),
            underlying_price=100.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# BacktestEngine with SETTLE_AT_EXPIRY policy
# ---------------------------------------------------------------------------


def _make_feature_df(timestamps: list, close_prices: list | None = None) -> pd.DataFrame:
    n = len(timestamps)
    prices = close_prices or [100.0] * n
    return pd.DataFrame(
        {
            "close": prices,
            "sigma": [0.01] * n,
            "S_D": [0.8] * n,
            "S_V": [-0.3] * n,
            "S_L": [0.7] * n,
            "S_G": [0.8] * n,
            "direction_regime": ["bull"] * n,
            "volatility_regime": ["low_vol"] * n,
            "liquidity_regime": ["high_liquidity"] * n,
            "dealer_flow_regime": ["supportive"] * n,
            "regime_key": ["r"] * n,
            "lower_1s": [90.0] * n,
            "upper_1s": [110.0] * n,
        },
        index=pd.to_datetime(timestamps),
    )


def _make_chain_df(timestamps: list, expiry: str, strike: float = 100.0) -> pd.DataFrame:
    rows = []
    for ts in timestamps:
        rows.append(
            {
                "timestamp": pd.Timestamp(ts),
                "underlying": "SPY",
                "expiry": pd.Timestamp(expiry),
                "option_type": "call",
                "strike": strike,
                "bid": 1.0,
                "ask": 1.2,
            }
        )
    return pd.DataFrame(rows)


class TestEngineExpiryLiquidationPolicy:
    def test_settle_at_expiry_produces_expiry_settlement_record(self, monkeypatch) -> None:
        timestamps = pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"])
        expiry = "2025-01-12"

        features = _make_feature_df(timestamps, close_prices=[100.0, 100.0, 115.0])
        chain = _make_chain_df(timestamps, expiry=expiry, strike=100.0)

        monkeypatch.setattr(
            "rlm.backtest.engine.decide_trade_for_bar",
            lambda row, **_: _single_leg_decision(expiry=expiry, side="long", strike=100.0),
        )
        monkeypatch.setattr(
            "rlm.backtest.engine.match_legs_to_chain",
            lambda decision, chain_slice: decision,
        )
        monkeypatch.setattr(
            "rlm.backtest.engine.select_nearest_expiry_slice",
            lambda row_chain, dte_min, dte_max: row_chain,
        )

        engine = BacktestEngine(
            lifecycle_config=LifecycleConfig(
                expiry_liquidation_policy=ExpiryLiquidationPolicy.SETTLE_AT_EXPIRY,
                close_at_expiry_if_open=True,
                force_close_dte=1,
                one_trade_per_bar=True,
            )
        )

        _, trades, _ = engine.run(features, chain)

        assert not trades.empty
        assert "expiry_settlement" in set(trades["exit_reason"])

    def test_liquidate_before_expiry_does_not_settle(self, monkeypatch) -> None:
        timestamps = pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"])
        expiry = "2025-01-12"

        features = _make_feature_df(timestamps)
        chain = _make_chain_df(timestamps, expiry=expiry)

        monkeypatch.setattr(
            "rlm.backtest.engine.decide_trade_for_bar",
            lambda row, **_: _single_leg_decision(expiry=expiry),
        )
        monkeypatch.setattr(
            "rlm.backtest.engine.match_legs_to_chain",
            lambda decision, chain_slice: decision,
        )
        monkeypatch.setattr(
            "rlm.backtest.engine.select_nearest_expiry_slice",
            lambda row_chain, dte_min, dte_max: row_chain,
        )

        engine = BacktestEngine(
            lifecycle_config=LifecycleConfig(
                expiry_liquidation_policy=ExpiryLiquidationPolicy.LIQUIDATE_BEFORE_EXPIRY,
                force_close_dte=1,
                close_at_expiry_if_open=True,
                one_trade_per_bar=True,
            )
        )

        _, trades, _ = engine.run(features, chain)

        assert not trades.empty
        # Should be forced_pre_expiry, not expiry_settlement
        assert "expiry_settlement" not in set(trades["exit_reason"])
        assert "forced_pre_expiry" in set(trades["exit_reason"])

    def test_commission_config_applied_on_open_and_close(self) -> None:
        """Hybrid commission model is applied correctly at entry and exit."""
        commission_config = CommissionConfig(
            model=CommissionModel.HYBRID,
            per_trade_rate=1.0,
            per_contract_rate=0.50,
        )
        lc = LifecycleConfig(commission_config=commission_config)
        portfolio = Portfolio(
            initial_capital=10_000.0,
            lifecycle_config=lc,
        )

        decision = _single_leg_decision(bid=5.0, ask=5.4)
        position_id = portfolio.open_from_decision(
            timestamp=pd.Timestamp("2025-01-10"),
            underlying_symbol="SPY",
            underlying_price=100.0,
            decision=decision,
            quantity=1,
        )
        assert position_id is not None

        pos = portfolio.open_positions[position_id]
        pos.current_exit_value_cache = pos.entry_cost + 20.0

        trade = portfolio.close_position(
            position_id=position_id,
            timestamp_exit=pd.Timestamp("2025-01-10"),
            underlying_price=100.0,
            exit_reason="test",
        )

        assert trade is not None
        # Commission at close: 1.0 + 0.50 * 1 leg * 1 qty = 1.50
        # exit_value = (entry_cost + 20.0) - 1.50
        expected_exit_value = pos.entry_cost + 20.0 - 1.50
        assert trade.exit_value == pytest.approx(expected_exit_value, abs=0.01)
