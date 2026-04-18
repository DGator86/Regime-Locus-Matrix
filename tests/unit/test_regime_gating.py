"""
Tests: summarize_by_regime + gate_regimes_by_expectancy prune bad regimes correctly.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from rlm.backtest.metrics import (
    compute_expectancy,
    gate_regimes_by_expectancy,
    summarize_by_regime,
)


def _trades(regime_pnls: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for regime, pnls in regime_pnls.items():
        for pnl in pnls:
            rows.append({"regime_key": regime, "pnl": pnl, "pnl_pct": pnl / 1000.0})
    return pd.DataFrame(rows)


def test_summarize_by_regime_returns_per_regime_stats() -> None:
    trades = _trades({
        "bull|low_vol": [100, 200, -50, 150],
        "bear|high_vol": [-200, -100, -300],
    })
    stats = summarize_by_regime(trades)
    assert "bull|low_vol" in stats
    assert "bear|high_vol" in stats

    bull = stats["bull|low_vol"]
    assert bull["trade_count"] == 4
    assert bull["win_rate"] == pytest.approx(0.75)
    assert bull["avg_return"] == pytest.approx(100.0)

    bear = stats["bear|high_vol"]
    assert bear["trade_count"] == 3
    assert bear["win_rate"] == pytest.approx(0.0)
    assert bear["avg_return"] < 0


def test_summarize_by_regime_computes_expectancy() -> None:
    trades = _trades({"range|low_vol": [100, -50]})
    stats = summarize_by_regime(trades)
    regime = stats["range|low_vol"]
    assert "expectancy" in regime
    # win_rate=0.5, avg_win=100, avg_loss=-50 → expectancy = 0.5×100 + 0.5×(-50) = 25
    assert regime["expectancy"] == pytest.approx(25.0)


def test_gate_regimes_by_expectancy_identifies_negative_regimes() -> None:
    trades = _trades({
        "bull|low_vol": [100, 200, 50],    # positive expectancy
        "bear|high_vol": [-200, -100],     # negative expectancy
        "range|low_vol": [10, -5],         # positive expectancy
    })
    stats = summarize_by_regime(trades)
    bad = gate_regimes_by_expectancy(stats, expectancy_floor=0.0)
    assert "bear|high_vol" in bad
    assert "bull|low_vol" not in bad
    assert "range|low_vol" not in bad


def test_gate_regimes_with_custom_floor() -> None:
    trades = _trades({
        "regime_a": [10, 5],   # expectancy ~ 7.5, above floor of 5
        "regime_b": [3, -1],   # expectancy ~ 1, below floor of 5
    })
    stats = summarize_by_regime(trades)
    bad = gate_regimes_by_expectancy(stats, expectancy_floor=5.0)
    assert "regime_b" in bad
    assert "regime_a" not in bad


def test_summarize_by_regime_empty_frame_returns_empty_dict() -> None:
    assert summarize_by_regime(pd.DataFrame()) == {}


def test_summarize_by_regime_missing_regime_col_returns_empty() -> None:
    trades = pd.DataFrame({"pnl": [100, -50]})
    assert summarize_by_regime(trades, regime_col="regime_key") == {}


def test_compute_expectancy_all_wins() -> None:
    pnls = pd.Series([100.0, 200.0, 150.0])
    exp = compute_expectancy(pnls)
    assert exp == pytest.approx(150.0)


def test_compute_expectancy_all_losses() -> None:
    pnls = pd.Series([-100.0, -200.0])
    exp = compute_expectancy(pnls)
    assert exp == pytest.approx(-150.0)


def test_compute_expectancy_empty() -> None:
    assert math.isnan(compute_expectancy(pd.Series([], dtype=float)))


def test_gate_returns_empty_set_when_all_profitable() -> None:
    trades = _trades({
        "regime_a": [100, 200, 300],
        "regime_b": [50, 75],
    })
    stats = summarize_by_regime(trades)
    bad = gate_regimes_by_expectancy(stats, expectancy_floor=0.0)
    assert len(bad) == 0
