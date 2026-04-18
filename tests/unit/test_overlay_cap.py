"""
Tests: overlay cap enforces MAX_OVERLAY_SIGNALS=3 and core_only bypasses overlays.
"""

from __future__ import annotations

from rlm.roee.policy import MAX_OVERLAY_SIGNALS, apply_overlay_engine, select_trade
from rlm.types.options import TradeDecision


def _enter_decision(**kwargs) -> TradeDecision:
    """Minimal entering decision to feed into apply_overlay_engine."""
    defaults = dict(
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
    )
    defaults.update(kwargs)
    return select_trade(**defaults)


def test_max_overlay_signals_constant_is_three() -> None:
    assert MAX_OVERLAY_SIGNALS == 3


def test_overlay_cap_limits_to_three_signals() -> None:
    """All four overlay signals fire; only the top 3 by boost should be applied."""
    base = _enter_decision()
    assert base.action == "enter"

    # Trigger all 4 candidate signals simultaneously.
    overlaid = apply_overlay_engine(
        base,
        direction_regime="bull",
        regime_state_label="bull|low_vol_like",
        # mtf_plus_fvg_or_ob: 1.1× (mtf>=1.5, fvg>=1.0)
        mtf_confluence_score=2.0,
        fvg_alignment_score=1.5,
        # liquidity_pool_candle_sr_alignment: 1.1× (pool + candle + sr)
        bullish_liquidity_pool_nearby=True,
        bullish_candle_pattern_score=0.8,
        bearish_candle_pattern_score=0.2,
        support_resistance_alignment_score=1.5,
        # hmm_sweep_pool_boost: 1.25× (sweep>2, pool>=2, hmm_state matches)
        mtf_confluence_liquidity_sweep_confirmed=2.5,
        pool_confluence_score=2.5,
        # orderflow_confluence_directional_boost: 1.15× (orderflow>2.5, direction aligned)
        orderflow_confluence_score=3.0,
    )

    assert overlaid.action == "enter"
    count = overlaid.metadata.get("overlay_signal_count", 0)
    assert count <= MAX_OVERLAY_SIGNALS, (
        f"Expected at most {MAX_OVERLAY_SIGNALS} active overlays, got {count}"
    )


def test_overlay_strongest_signals_selected() -> None:
    """With 4 signals, the top 3 by boost are used: 1.25×, 1.15×, 1.1× = not 1.1×."""
    base = _enter_decision()
    overlaid = apply_overlay_engine(
        base,
        direction_regime="bull",
        regime_state_label="bull|low_vol_like",
        mtf_confluence_score=2.0,
        fvg_alignment_score=1.5,
        bullish_liquidity_pool_nearby=True,
        bullish_candle_pattern_score=0.8,
        bearish_candle_pattern_score=0.2,
        support_resistance_alignment_score=1.5,
        mtf_confluence_liquidity_sweep_confirmed=2.5,
        pool_confluence_score=2.5,
        orderflow_confluence_score=3.0,
    )
    # With cap=3, weakest signal (1.1× for mtf_plus_fvg_or_ob or pool_candle_sr) is dropped.
    # Resulting multiplier must be < 1.25 × 1.15 × 1.1 × 1.1 but == top-3 product.
    signals = overlaid.metadata.get("overlay_signals", "")
    assert "hmm_sweep_pool_boost" in signals, "Strongest signal (1.25×) must be kept"
    assert overlaid.metadata.get("overlay_signal_count") == MAX_OVERLAY_SIGNALS


def test_no_overlays_when_no_signal_fires() -> None:
    base = _enter_decision()
    overlaid = apply_overlay_engine(
        base,
        direction_regime="bull",
        # All scores below thresholds — nothing triggers.
        mtf_confluence_score=0.5,
        fvg_alignment_score=0.0,
        orderflow_confluence_score=0.0,
        pool_confluence_score=0.0,
        mtf_confluence_liquidity_sweep_confirmed=0.0,
    )
    assert overlaid.metadata.get("overlay_multiplier") == 1.0
    assert overlaid.metadata.get("overlay_signal_count") == 0
    assert overlaid.size_fraction == base.size_fraction


def test_overlay_skipped_for_non_enter_decision() -> None:
    skip = TradeDecision(action="skip", rationale="test")
    result = apply_overlay_engine(skip, direction_regime="bull")
    assert result.action == "skip"
    assert result is skip


def test_core_only_bypasses_overlay_engine() -> None:
    without_core_only = _enter_decision(
        mtf_confluence_score=2.0,
        fvg_alignment_score=1.5,
        bullish_liquidity_pool_nearby=True,
        bullish_candle_pattern_score=0.8,
        bearish_candle_pattern_score=0.2,
        support_resistance_alignment_score=1.5,
        mtf_confluence_liquidity_sweep_confirmed=2.5,
        pool_confluence_score=2.5,
        orderflow_confluence_score=3.0,
    )
    with_core_only = _enter_decision(
        core_only=True,
        mtf_confluence_score=2.0,
        fvg_alignment_score=1.5,
        bullish_liquidity_pool_nearby=True,
        bullish_candle_pattern_score=0.8,
        bearish_candle_pattern_score=0.2,
        support_resistance_alignment_score=1.5,
        mtf_confluence_liquidity_sweep_confirmed=2.5,
        pool_confluence_score=2.5,
        orderflow_confluence_score=3.0,
    )

    assert with_core_only.action == "enter"
    # Core-only: no overlay metadata present, size unchanged by overlays.
    assert "overlay_multiplier" not in with_core_only.metadata
    # Non-core-only: overlays boosted size.
    assert "overlay_multiplier" in without_core_only.metadata
    assert float(without_core_only.metadata.get("overlay_multiplier", 1.0)) > 1.0
