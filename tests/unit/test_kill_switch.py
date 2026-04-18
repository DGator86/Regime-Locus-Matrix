"""
Tests: KillSwitch fires on drawdown, vol anomaly, and consecutive losses.
"""

from __future__ import annotations

import pytest

from rlm.backtest.kill_switch import KillSwitch


def _fresh(
    max_drawdown: float = -0.15,
    anomaly_vol_multiplier: float | None = 3.0,
    consecutive_loss_limit: int | None = 5,
    initial_equity: float = 100_000.0,
) -> KillSwitch:
    ks = KillSwitch(
        max_drawdown=max_drawdown,
        anomaly_vol_multiplier=anomaly_vol_multiplier,
        consecutive_loss_limit=consecutive_loss_limit,
    )
    ks.reset(initial_equity)
    return ks


# ---------------------------------------------------------------------------
# Drawdown gate
# ---------------------------------------------------------------------------

def test_not_halted_initially() -> None:
    ks = _fresh()
    assert not ks.halted


def test_no_halt_within_drawdown_limit() -> None:
    ks = _fresh(max_drawdown=-0.15, initial_equity=100_000.0)
    ks.update(current_equity=90_000.0)   # -10% drawdown — within -15% limit
    assert not ks.halted


def test_halt_on_drawdown_breach() -> None:
    ks = _fresh(max_drawdown=-0.15, initial_equity=100_000.0)
    triggered = ks.update(current_equity=84_000.0)  # -16% drawdown
    assert triggered is True
    assert ks.halted
    assert "drawdown" in ks.halt_reason.lower()


def test_high_water_mark_updates_on_equity_rise() -> None:
    ks = _fresh(max_drawdown=-0.20, initial_equity=100_000.0)
    ks.update(current_equity=120_000.0)   # new HWM
    ks.update(current_equity=110_000.0)   # -8.3% from 120k — fine
    assert not ks.halted
    triggered = ks.update(current_equity=95_000.0)  # -20.8% from 120k — breach
    assert triggered is True


def test_already_halted_returns_true_without_re_eval() -> None:
    ks = _fresh(max_drawdown=-0.10, initial_equity=100_000.0)
    ks.update(current_equity=85_000.0)   # breach
    assert ks.halted
    # Further calls still return True
    result = ks.update(current_equity=99_000.0)
    assert result is True


# ---------------------------------------------------------------------------
# Volatility anomaly gate
# ---------------------------------------------------------------------------

def test_no_halt_when_vol_normal() -> None:
    ks = _fresh(anomaly_vol_multiplier=3.0)
    ks.update(current_equity=100_000.0, realized_vol=0.15)
    ks.update(current_equity=100_000.0, realized_vol=0.16)
    ks.update(current_equity=100_000.0, realized_vol=0.17)
    assert not ks.halted


def test_halt_on_vol_spike() -> None:
    ks = _fresh(anomaly_vol_multiplier=3.0, initial_equity=100_000.0)
    # Build stable baseline vol of ~0.15
    for _ in range(10):
        ks.update(current_equity=100_000.0, realized_vol=0.15)
    # Spike vol to 2× baseline — should not trigger at 3× multiplier
    ks.update(current_equity=100_000.0, realized_vol=0.30)
    assert not ks.halted
    # Spike vol to 5× baseline — should trigger
    triggered = ks.update(current_equity=100_000.0, realized_vol=0.75)
    assert triggered is True
    assert ks.halted
    assert "vol" in ks.halt_reason.lower()


def test_vol_anomaly_disabled_when_multiplier_is_none() -> None:
    ks = _fresh(anomaly_vol_multiplier=None, initial_equity=100_000.0)
    ks.update(current_equity=100_000.0, realized_vol=5.0)  # extreme vol
    assert not ks.halted


# ---------------------------------------------------------------------------
# Consecutive loss gate
# ---------------------------------------------------------------------------

def test_no_halt_below_consecutive_loss_limit() -> None:
    ks = _fresh(consecutive_loss_limit=5, initial_equity=100_000.0)
    for _ in range(4):
        ks.update(current_equity=100_000.0, last_trade_pnl=-100.0)
    assert not ks.halted


def test_halt_on_consecutive_loss_limit() -> None:
    ks = _fresh(consecutive_loss_limit=5, initial_equity=100_000.0)
    for _ in range(4):
        ks.update(current_equity=100_000.0, last_trade_pnl=-100.0)
    triggered = ks.update(current_equity=100_000.0, last_trade_pnl=-50.0)
    assert triggered is True
    assert ks.halted
    assert "consecutive" in ks.halt_reason.lower()


def test_win_resets_consecutive_loss_counter() -> None:
    ks = _fresh(consecutive_loss_limit=5, initial_equity=100_000.0)
    for _ in range(4):
        ks.update(current_equity=100_000.0, last_trade_pnl=-100.0)
    ks.update(current_equity=100_000.0, last_trade_pnl=200.0)  # win resets counter
    for _ in range(4):
        ks.update(current_equity=100_000.0, last_trade_pnl=-100.0)
    assert not ks.halted  # still at 4 consecutive losses


def test_consecutive_loss_disabled_when_none() -> None:
    ks = _fresh(consecutive_loss_limit=None, initial_equity=100_000.0)
    for _ in range(20):
        ks.update(current_equity=100_000.0, last_trade_pnl=-100.0)
    assert not ks.halted


# ---------------------------------------------------------------------------
# Reset behaviour
# ---------------------------------------------------------------------------

def test_reset_clears_halted_state() -> None:
    ks = _fresh(max_drawdown=-0.10, initial_equity=100_000.0)
    ks.update(current_equity=85_000.0)  # breach
    assert ks.halted
    ks.reset(initial_equity=100_000.0)
    assert not ks.halted
    assert ks.halt_reason == ""
