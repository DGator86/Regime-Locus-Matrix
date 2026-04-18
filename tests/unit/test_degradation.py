"""
Tests: DegradationAlert transitions correctly between none / reduce_size / halt.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from rlm.backtest.degradation import (
    DegradationThresholds,
    check_degradation,
    compute_rolling_metrics,
)


def _trades(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnls, "pnl_pct": [p / 1000.0 for p in pnls]})


def _thresholds(**kwargs) -> DegradationThresholds:
    defaults = dict(
        expectancy_floor=0.0,
        winrate_floor=0.3,
        drawdown_floor=-0.15,
        min_trades=3,
    )
    defaults.update(kwargs)
    return DegradationThresholds(**defaults)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def test_compute_rolling_metrics_produces_expected_columns() -> None:
    trades = _trades([100, -50, 200, -30, 150])
    rolling = compute_rolling_metrics(trades, window=3)
    for col in ("rolling_expectancy", "rolling_winrate", "rolling_drawdown", "rolling_avg_return"):
        assert col in rolling.columns


def test_rolling_expectancy_is_rolling_mean() -> None:
    trades = _trades([100, 100, 100])
    rolling = compute_rolling_metrics(trades, window=3)
    assert rolling["rolling_expectancy"].iloc[-1] == pytest.approx(100.0)


def test_rolling_winrate_correct() -> None:
    trades = _trades([100, -50, 100])
    rolling = compute_rolling_metrics(trades, window=3)
    # 2 wins out of 3 = 0.667
    assert rolling["rolling_winrate"].iloc[-1] == pytest.approx(2 / 3, rel=1e-4)


def test_compute_rolling_metrics_empty_returns_empty() -> None:
    result = compute_rolling_metrics(pd.DataFrame())
    assert result.empty


# ---------------------------------------------------------------------------
# check_degradation alert levels
# ---------------------------------------------------------------------------

def test_no_alert_when_healthy() -> None:
    trades = _trades([100, 200, 150, 80, 120])
    rolling = compute_rolling_metrics(trades, window=5)
    alert = check_degradation(rolling, _thresholds())
    assert alert.degraded is False
    assert alert.recommended_action == "none"
    assert len(alert.reasons) == 0


def test_reduce_size_on_single_threshold_breach() -> None:
    # Negative expectancy only → one breach → reduce_size
    trades = _trades([-100, -200, -50, -150, -80])
    rolling = compute_rolling_metrics(trades, window=5)
    alert = check_degradation(rolling, _thresholds(expectancy_floor=0.0, winrate_floor=0.0, drawdown_floor=-1.0))
    assert alert.degraded is True
    assert alert.recommended_action == "reduce_size"
    assert len(alert.reasons) == 1


def test_halt_on_multiple_threshold_breaches() -> None:
    # Both expectancy and win rate breached → halt
    trades = _trades([-100, -200, -50])
    rolling = compute_rolling_metrics(trades, window=3)
    alert = check_degradation(rolling, _thresholds(
        expectancy_floor=0.0,
        winrate_floor=0.5,  # 0 wins out of 3 breaches this
        drawdown_floor=-1.0,
    ))
    assert alert.degraded is True
    assert alert.recommended_action == "halt"
    assert len(alert.reasons) >= 2


def test_no_alert_below_min_trades() -> None:
    trades = _trades([100, -50])  # only 2 trades
    rolling = compute_rolling_metrics(trades, window=5)
    alert = check_degradation(rolling, _thresholds(min_trades=10))
    assert alert.degraded is False
    assert alert.recommended_action == "none"
    assert "Insufficient" in alert.reasons[0]


def test_empty_rolling_returns_clean_alert() -> None:
    alert = check_degradation(pd.DataFrame())
    assert alert.degraded is False
    assert alert.recommended_action == "none"
    assert alert.trade_count == 0


def test_alert_contains_rolling_values() -> None:
    trades = _trades([100, -50, 200, -30])
    rolling = compute_rolling_metrics(trades, window=4)
    alert = check_degradation(rolling, _thresholds())
    assert math.isfinite(alert.rolling_expectancy)
    assert math.isfinite(alert.rolling_winrate)
    assert alert.trade_count == 4


def test_reasons_include_descriptive_text() -> None:
    trades = _trades([-200, -300, -100])
    rolling = compute_rolling_metrics(trades, window=3)
    alert = check_degradation(rolling, _thresholds(expectancy_floor=0.0, winrate_floor=0.5))
    for reason in alert.reasons:
        assert len(reason) > 5, "Each reason should be a descriptive string"
