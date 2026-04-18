"""
Tests: is_tradeable_environment gates all four no-trade conditions.
"""

from __future__ import annotations

import pytest

from rlm.roee.risk import is_tradeable_environment
from rlm.roee.policy import select_trade


def test_passes_when_all_conditions_clean() -> None:
    ok, reason = is_tradeable_environment(
        has_major_event=False,
        bid_ask_spread_pct=0.01,
        volume_ratio=1.5,
        regime_transition=False,
    )
    assert ok is True
    assert reason == ""


def test_blocks_on_major_event() -> None:
    ok, reason = is_tradeable_environment(has_major_event=True)
    assert ok is False
    assert "event" in reason.lower()


def test_blocks_on_wide_spread() -> None:
    ok, reason = is_tradeable_environment(bid_ask_spread_pct=0.10, max_spread_pct=0.05)
    assert ok is False
    assert "spread" in reason.lower()


def test_passes_at_exact_spread_limit() -> None:
    ok, _ = is_tradeable_environment(bid_ask_spread_pct=0.05, max_spread_pct=0.05)
    assert ok is True


def test_blocks_on_low_volume() -> None:
    ok, reason = is_tradeable_environment(volume_ratio=0.3, min_volume_ratio=0.5)
    assert ok is False
    assert "volume" in reason.lower()


def test_passes_at_exact_volume_floor() -> None:
    ok, _ = is_tradeable_environment(volume_ratio=0.5, min_volume_ratio=0.5)
    assert ok is True


def test_blocks_on_regime_transition() -> None:
    ok, reason = is_tradeable_environment(regime_transition=True)
    assert ok is False
    assert "transition" in reason.lower()


def test_event_check_fires_before_spread() -> None:
    """Event gate has priority — reason should mention event, not spread."""
    ok, reason = is_tradeable_environment(
        has_major_event=True,
        bid_ask_spread_pct=0.20,
    )
    assert ok is False
    assert "event" in reason.lower()


def test_select_trade_respects_no_trade_zone_volume() -> None:
    """volume_ratio gate wired into select_trade blocks entry."""
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
        strike_increment=5.0,
        volume_ratio=0.2,  # well below 0.5 threshold
    )
    assert decision.action == "skip"
    assert "volume" in decision.rationale.lower()


def test_select_trade_respects_no_trade_zone_transition() -> None:
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
        strike_increment=5.0,
        regime_transition=True,
    )
    assert decision.action == "skip"
    assert "transition" in decision.rationale.lower()


def test_no_trade_zone_passes_with_none_inputs() -> None:
    """None volume/spread should not trigger gates."""
    ok, reason = is_tradeable_environment(
        has_major_event=False,
        bid_ask_spread_pct=None,
        volume_ratio=None,
        regime_transition=False,
    )
    assert ok is True
    assert reason == ""
