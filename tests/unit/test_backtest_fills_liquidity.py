import pandas as pd
import pytest

from rlm.backtest.engine import BacktestEngine
from rlm.backtest.fills import FillConfig, entry_fill_price
from rlm.backtest.lifecycle import LifecycleConfig
from rlm.backtest.slippage import SlippageConfig
from rlm.types.options import OptionLeg, TradeCandidate, TradeDecision


def test_entry_fill_price_applies_liquidity_impact_when_size_exceeds_depth() -> None:
    cfg = FillConfig(liquidity_impact_factor=1.0)
    small = entry_fill_price(side="long", bid=1.0, ask=1.2, config=cfg, quantity=1, quote_size=10)
    large = entry_fill_price(side="long", bid=1.0, ask=1.2, config=cfg, quantity=50, quote_size=10)
    assert large > small


def _features(timestamps: pd.DatetimeIndex, realized_vols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0] * len(timestamps),
            "sigma": [0.01] * len(timestamps),
            "S_D": [0.8] * len(timestamps),
            "S_V": [-0.5] * len(timestamps),
            "S_L": [0.7] * len(timestamps),
            "S_G": [0.8] * len(timestamps),
            "lower_1s": [50.0] * len(timestamps),
            "upper_1s": [150.0] * len(timestamps),
            "realized_vol": realized_vols,
        },
        index=timestamps,
    )


def _chain(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "underlying": "SPY",
                "expiry": pd.Timestamp("2025-02-14"),
                "option_type": "call",
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.2,
            }
            for ts in timestamps
        ]
    )


def _long_call_decision() -> TradeDecision:
    candidate = TradeCandidate(
        strategy_name="long_call",
        regime_key="bull|low_vol|high_liquidity|supportive",
        rationale="test",
        target_dte_min=14,
        target_dte_max=45,
        target_profit_pct=10.0,
        max_risk_pct=0.02,
        long_sigma=0.0,
    )
    return TradeDecision(
        action="enter",
        strategy_name="long_call",
        regime_key=candidate.regime_key,
        rationale="test",
        target_profit_pct=10.0,
        max_risk_pct=0.02,
        candidate=candidate,
        legs=[OptionLeg(side="long", option_type="call", strike=100.0)],
    )


def _vol_sensitive_fill_config() -> FillConfig:
    return FillConfig(
        slippage=SlippageConfig(
            spread_fraction=0.25,
            min_slippage=0.0,
            per_contract_flat=0.0,
            vol_sensitivity=3.0,
        )
    )


def test_backtest_entry_fills_use_bar_realized_vol(monkeypatch) -> None:
    timestamps = pd.date_range("2025-01-10", periods=1, freq="D")

    monkeypatch.setattr("rlm.backtest.engine.decide_trade_for_bar", lambda row, **_: _long_call_decision())

    engine = BacktestEngine(
        initial_capital=10_000.0,
        fill_config=_vol_sensitive_fill_config(),
        lifecycle_config=LifecycleConfig(commission_per_contract=0.0),
    )
    engine.run(_features(timestamps, [1.0]), _chain(timestamps))

    position = next(iter(engine.portfolio.open_positions.values()))
    assert position.entry_cost == pytest.approx(140.0)
    assert engine.portfolio.cash == pytest.approx(9_860.0)


def test_backtest_exit_repricing_uses_bar_realized_vol(monkeypatch) -> None:
    timestamps = pd.date_range("2025-01-10", periods=2, freq="D")
    decisions = iter([_long_call_decision(), TradeDecision(action="hold")])

    monkeypatch.setattr("rlm.backtest.engine.decide_trade_for_bar", lambda row, **_: next(decisions))

    engine = BacktestEngine(
        initial_capital=10_000.0,
        fill_config=_vol_sensitive_fill_config(),
        lifecycle_config=LifecycleConfig(commission_per_contract=0.0, max_holding_bars=1),
    )
    _, trades, _ = engine.run(_features(timestamps, [0.0, 1.0]), _chain(timestamps))

    assert trades.iloc[0]["pnl"] == pytest.approx(-45.0)
    assert engine.portfolio.cash == pytest.approx(9_955.0)
