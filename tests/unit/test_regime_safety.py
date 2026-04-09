import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.roee.decision import select_trade_for_row
from rlm.roee.pipeline import ROEEConfig, apply_roee_policy
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.types.options import TradeDecision


def _bull_row(regime_key: str = "bull|low_vol|high_liquidity|supportive") -> dict[str, object]:
    return {
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
        "regime_key": regime_key,
        "lower_1s": 4950.0,
        "upper_1s": 5050.0,
    }


def test_attach_regime_safety_columns_respects_purge_gap() -> None:
    df = pd.DataFrame(
        {
            "regime_key": [
                "bull|low_vol|high_liquidity|supportive",
                "bull|low_vol|high_liquidity|supportive",
                "bull|low_vol|high_liquidity|supportive",
                "range|low_vol|high_liquidity|supportive",
                "range|low_vol|high_liquidity|supportive",
                "range|low_vol|high_liquidity|supportive",
            ]
        }
    )

    no_purge = attach_regime_safety_columns(df, min_regime_train_samples=2, purge_bars=0)
    with_purge = attach_regime_safety_columns(df, min_regime_train_samples=2, purge_bars=2)

    assert int(no_purge.iloc[-1]["regime_train_sample_count"]) == 2
    assert bool(no_purge.iloc[-1]["regime_safety_ok"]) is True
    assert int(with_purge.iloc[-1]["regime_train_sample_count"]) == 0
    assert bool(with_purge.iloc[-1]["regime_safety_ok"]) is False


def test_select_trade_for_row_holds_when_regime_is_undertrained() -> None:
    row = pd.Series(_bull_row())

    decision = select_trade_for_row(
        row,
        strike_increment=5.0,
        regime_train_sample_count=3,
        min_regime_train_samples=5,
        regime_purge_bars=2,
    )

    assert decision.action == "hold"
    assert decision.strategy_name == "regime_safety_check"
    assert decision.metadata["regime_safety_ok"] is False
    assert decision.metadata["regime_train_sample_count"] == 3


def test_apply_roee_policy_blocks_unfamiliar_regime() -> None:
    df = pd.DataFrame(
        [
            _bull_row("bull|low_vol|high_liquidity|supportive"),
            _bull_row("bull|low_vol|high_liquidity|supportive"),
            _bull_row("range|low_vol|high_liquidity|supportive")
            | {
                "direction_regime": "range",
                "S_D": 0.0,
            },
        ]
    )

    out = apply_roee_policy(
        df,
        strike_increment=5.0,
        config=ROEEConfig(min_regime_train_samples=2, purge_bars=0),
    )

    assert "regime_train_sample_count" in out.columns
    assert "regime_safety_ok" in out.columns
    assert out.iloc[-1]["roee_action"] == "hold"
    assert out.iloc[-1]["roee_strategy"] == "regime_safety_check"
    assert bool(out.iloc[-1]["regime_safety_ok"]) is False


def test_engine_passes_regime_safety_counts_to_decision(monkeypatch) -> None:
    timestamps = pd.date_range("2025-01-10", periods=3, freq="D")
    features = pd.DataFrame([_bull_row()] * 3, index=timestamps)
    chain = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "underlying": "SPY",
                "expiry": pd.Timestamp("2025-02-14"),
                "option_type": "call",
                "strike": 5000.0,
                "bid": 8.0,
                "ask": 8.4,
            }
            for ts in timestamps
        ]
    )
    seen_counts: list[int] = []
    seen_mins: list[int] = []

    def _capture_decision(row: pd.Series, **kwargs: object) -> TradeDecision:
        seen_counts.append(int(kwargs["regime_train_sample_count"]))
        seen_mins.append(int(kwargs["min_regime_train_samples"]))
        return TradeDecision(action="hold", regime_key=str(row["regime_key"]))

    monkeypatch.setattr("rlm.backtest.engine.decide_trade_for_bar", _capture_decision)

    engine = BacktestEngine(roee_config=ROEEConfig(min_regime_train_samples=2, purge_bars=0))
    engine.run(features, chain)

    assert seen_counts == [0, 1, 2]
    assert seen_mins == [2, 2, 2]
