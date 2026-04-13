from __future__ import annotations

import math

from rlm.roee.risk import (
    should_require_defined_risk,
    should_skip_for_event_risk,
    spread_quality_ok,
)
from rlm.roee.sizing import (
    apply_uncertainty_vault,
    compute_confidence,
    compute_regime_adjusted_kelly_fraction,
    compute_regime_penalty_multiplier,
    compute_size_fraction,
    kelly_voltarget_size,
    parse_latent_regime_label,
    quantize_fraction,
)
from rlm.roee.strategy_map import get_strategy_for_regime
from rlm.roee.strike_selection import build_legs_from_candidate
from rlm.types.options import TradeDecision


def select_trade(
    *,
    current_price: float,
    sigma: float,
    s_d: float,
    s_v: float,
    s_l: float,
    s_g: float,
    direction_regime: str,
    volatility_regime: str,
    liquidity_regime: str,
    dealer_flow_regime: str,
    regime_key: str,
    bid_ask_spread_pct: float | None = None,
    has_major_event: bool = False,
    strike_increment: float = 1.0,
    short_dte: bool = False,
    forecast_return: float | None = None,
    forecast_uncertainty: float | None = None,
    realized_vol: float | None = None,
    use_dynamic_sizing: bool = False,
    vol_target: float = 0.15,
    max_kelly_fraction: float = 0.25,
    max_capital_fraction: float = 0.5,
    regime_adjusted_kelly: bool = True,
    regime_state_label: str | None = None,
    regime_state_confidence: float | None = None,
    high_vol_kelly_multiplier: float = 0.5,
    transition_kelly_multiplier: float = 0.75,
    calm_trend_kelly_multiplier: float = 1.25,
    vault_uncertainty_threshold: float | None = 0.03,
    vault_size_multiplier: float = 0.5,
    mtf_confluence_score: float | None = None,
    mtf_confluence_liquidity_sweep_confirmed: float | None = None,
    pool_confluence_score: float | None = None,
    orderflow_confluence_score: float | None = None,
    bullish_liquidity_pool_nearby: bool = False,
    bearish_liquidity_pool_nearby: bool = False,
    fvg_alignment_score: float | None = None,
    order_block_alignment_score: float | None = None,
    bullish_candle_pattern_score: float | None = None,
    bearish_candle_pattern_score: float | None = None,
    support_resistance_alignment_score: float | None = None,
) -> TradeDecision:
    """
    Main ROEE entry point for one bar / one underlying snapshot.

    Parameters
    ----------
    short_dte:
        When True, the strategy map selects 0DTE / 1DTE intraday strategies
        instead of the default 14–60 DTE multi-week structures.
    """
    if not math.isfinite(current_price) or current_price <= 0:
        return TradeDecision(action="skip", rationale="Invalid current price.")

    if not math.isfinite(sigma) or sigma <= 0:
        return TradeDecision(action="skip", rationale="Invalid sigma.")

    if should_skip_for_event_risk(has_major_event):
        return TradeDecision(action="skip", rationale="Major event risk filter active.")

    if not spread_quality_ok(bid_ask_spread_pct):
        return TradeDecision(action="skip", rationale="Spread quality filter failed.")

    candidate = get_strategy_for_regime(
        direction=direction_regime,
        volatility=volatility_regime,
        liquidity=liquidity_regime,
        dealer_flow=dealer_flow_regime,
        short_dte=short_dte,
    )

    if candidate.strategy_name == "no_trade_or_micro_position":
        return TradeDecision(
            action="skip",
            strategy_name=candidate.strategy_name,
            regime_key=regime_key,
            rationale=candidate.rationale,
            candidate=candidate,
        )

    require_defined_risk = should_require_defined_risk(s_l, s_g)
    if require_defined_risk and not candidate.defined_risk:
        return TradeDecision(
            action="skip",
            strategy_name=candidate.strategy_name,
            regime_key=regime_key,
            rationale="Candidate rejected: defined-risk required.",
            candidate=candidate,
        )

    confidence = compute_confidence(s_d=s_d, s_v=s_v, s_l=s_l, s_g=s_g)
    size_fraction = compute_size_fraction(
        confidence=confidence,
        base_risk_pct=candidate.max_risk_pct,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        direction_regime=direction_regime,
    )
    size_model = "confidence"
    sizing_meta: dict[str, float | str | bool | None] = {
        "confidence": confidence,
        "require_defined_risk": require_defined_risk,
        "current_price": float(current_price),
        "sigma": float(sigma),
    }

    forecast_ok = forecast_return is not None and math.isfinite(forecast_return)
    vol_ok = realized_vol is not None and math.isfinite(realized_vol) and realized_vol > 0.0
    if use_dynamic_sizing and forecast_ok and vol_ok:
        effective_kelly_fraction = float(max_kelly_fraction)
        kelly_direction_regime, kelly_volatility_regime = parse_latent_regime_label(
            regime_state_label
        )
        if regime_adjusted_kelly:
            effective_kelly_fraction = compute_regime_adjusted_kelly_fraction(
                base_kelly_fraction=float(max_kelly_fraction),
                regime_state_label=regime_state_label,
                regime_state_confidence=regime_state_confidence,
                high_vol_multiplier=high_vol_kelly_multiplier,
                transition_multiplier=transition_kelly_multiplier,
                calm_trend_multiplier=calm_trend_kelly_multiplier,
            )
        regime_penalty = compute_regime_penalty_multiplier(
            confidence=confidence,
            base_risk_pct=candidate.max_risk_pct,
            liquidity_regime=liquidity_regime,
            dealer_flow_regime=dealer_flow_regime,
            direction_regime=direction_regime,
        )
        raw_dynamic_size = kelly_voltarget_size(
            forecast_return=float(forecast_return),
            realized_vol=float(realized_vol),
            vol_target=vol_target,
            max_kelly_fraction=effective_kelly_fraction,
            max_capital_fraction=max_capital_fraction,
        )
        size_fraction = quantize_fraction(
            min(
                candidate.max_risk_pct,
                max_capital_fraction,
                raw_dynamic_size * regime_penalty,
            )
        )
        size_model = "kelly_vol_target"
        sizing_meta.update(
            {
                "forecast_return": float(forecast_return),
                "realized_vol": float(realized_vol),
                "regime_penalty": regime_penalty,
                "vol_target": float(vol_target),
                "base_kelly_fraction": float(max_kelly_fraction),
                "max_kelly_fraction": effective_kelly_fraction,
                "max_capital_fraction": float(max_capital_fraction),
                "raw_dynamic_size": raw_dynamic_size,
                "regime_adjusted_kelly": regime_adjusted_kelly,
                "regime_state_label": regime_state_label or "",
                "regime_state_confidence": (
                    float(regime_state_confidence)
                    if regime_state_confidence is not None
                    and math.isfinite(float(regime_state_confidence))
                    else None
                ),
                "kelly_regime_source": "latent_state" if regime_state_label else "disabled",
                "kelly_direction_regime": kelly_direction_regime or "",
                "kelly_volatility_regime": kelly_volatility_regime or "",
                "kelly_fraction_multiplier": (
                    quantize_fraction(effective_kelly_fraction / float(max_kelly_fraction))
                    if float(max_kelly_fraction) > 0.0
                    else 0.0
                ),
            }
        )

    size_fraction, vault_meta = apply_uncertainty_vault(
        size_fraction=size_fraction,
        forecast_uncertainty=forecast_uncertainty,
        uncertainty_threshold=vault_uncertainty_threshold,
        size_multiplier=vault_size_multiplier,
    )
    sizing_meta.update(vault_meta)

    # Example MTF + market-structure overlays for newly surfaced features:
    # - MTF confluence / liquidity sweep confirmation
    # - liquidity pools
    # - fair value gaps (FVG)
    # - order blocks
    # - candle patterns
    # - support / resistance context
    #
    # These rules are intentionally additive and conservative:
    # 1) build an overlay multiplier from independent confirmation signals;
    # 2) only apply the strongest boost under the "correct" HMM state;
    # 3) clamp by max risk/capital controls after applying the multiplier.
    overlay_multiplier = 1.0
    overlay_signals: list[str] = []
    hmm_direction_regime, hmm_volatility_regime = parse_latent_regime_label(regime_state_label)

    mtf_score = float(mtf_confluence_score) if mtf_confluence_score is not None else 0.0
    sweep_score = (
        float(mtf_confluence_liquidity_sweep_confirmed)
        if mtf_confluence_liquidity_sweep_confirmed is not None
        else 0.0
    )
    pool_score = float(pool_confluence_score) if pool_confluence_score is not None else 0.0
    orderflow_score = (
        float(orderflow_confluence_score) if orderflow_confluence_score is not None else 0.0
    )
    fvg_score = float(fvg_alignment_score) if fvg_alignment_score is not None else 0.0
    ob_score = (
        float(order_block_alignment_score) if order_block_alignment_score is not None else 0.0
    )
    sr_score = (
        float(support_resistance_alignment_score)
        if support_resistance_alignment_score is not None
        else 0.0
    )
    bullish_candle = (
        float(bullish_candle_pattern_score) if bullish_candle_pattern_score is not None else 0.0
    )
    bearish_candle = (
        float(bearish_candle_pattern_score) if bearish_candle_pattern_score is not None else 0.0
    )

    # Example directional structure filters:
    # - Bull setups: require bullish pools + bullish candle bias + structure support.
    # - Bear setups: require bearish pools + bearish candle bias + structure resistance.
    bullish_structure_confirmed = (
        (direction_regime == "bull" or hmm_direction_regime == "bull")
        and bullish_liquidity_pool_nearby
        and bullish_candle > bearish_candle
        and sr_score >= 1.0
    )
    bearish_structure_confirmed = (
        (direction_regime == "bear" or hmm_direction_regime == "bear")
        and bearish_liquidity_pool_nearby
        and bearish_candle > bullish_candle
        and sr_score >= 1.0
    )

    if mtf_score >= 1.5 and (fvg_score >= 1.0 or ob_score >= 1.0):
        overlay_multiplier *= 1.1
        overlay_signals.append("mtf_plus_fvg_or_ob")

    if bullish_structure_confirmed or bearish_structure_confirmed:
        overlay_multiplier *= 1.1
        overlay_signals.append("liquidity_pool_candle_sr_alignment")

    # Explicit request:
    # boost size_multiplier when:
    #   mtf_confluence_liquidity_sweep_confirmed > 2.0
    #   and pool_confluence_score >= 2
    # inside the correct HMM state (trend + non-high-vol context).
    hmm_state_allows_sweep_boost = (
        hmm_direction_regime == direction_regime and hmm_direction_regime in {"bull", "bear"}
    ) and hmm_volatility_regime not in {"high", "crisis", "panic"}
    if sweep_score > 2.0 and pool_score >= 2.0 and hmm_state_allows_sweep_boost:
        overlay_multiplier *= 1.25
        overlay_signals.append("hmm_sweep_pool_boost")

    # Conservative order-flow add-on:
    # - requires a strong substructure cluster (orderflow_confluence_score > 2.5),
    # - requires directional alignment with the active trend/HMM direction,
    # - applies only a modest +15% multiplier so core risk controls remain dominant.
    orderflow_direction_aligned = direction_regime in {"bull", "bear"} and (
        hmm_direction_regime in {"", direction_regime} or hmm_direction_regime == direction_regime
    )
    if orderflow_score > 2.5 and orderflow_direction_aligned:
        overlay_multiplier *= 1.15
        overlay_signals.append("orderflow_confluence_directional_boost")

    if overlay_multiplier != 1.0:
        size_fraction = quantize_fraction(
            min(candidate.max_risk_pct, max_capital_fraction, size_fraction * overlay_multiplier)
        )
    sizing_meta.update(
        {
            "mtf_overlay_multiplier": overlay_multiplier,
            "mtf_overlay_signals": "|".join(overlay_signals),
            "mtf_confluence_score": mtf_score,
            "mtf_confluence_liquidity_sweep_confirmed": sweep_score,
            "pool_confluence_score": pool_score,
            "orderflow_confluence_score": orderflow_score,
            "fvg_alignment_score": fvg_score,
            "order_block_alignment_score": ob_score,
            "support_resistance_alignment_score": sr_score,
            "bullish_candle_pattern_score": bullish_candle,
            "bearish_candle_pattern_score": bearish_candle,
            "hmm_state_allows_sweep_boost": hmm_state_allows_sweep_boost,
            "orderflow_direction_aligned": orderflow_direction_aligned,
        }
    )

    if size_fraction <= 0:
        return TradeDecision(
            action="skip",
            strategy_name=candidate.strategy_name,
            regime_key=regime_key,
            rationale="Position size reduced to zero by risk controls.",
            candidate=candidate,
        )

    legs = build_legs_from_candidate(
        candidate=candidate,
        current_price=current_price,
        sigma=sigma,
        strike_increment=strike_increment,
    )

    return TradeDecision(
        action="enter",
        strategy_name=candidate.strategy_name,
        regime_key=regime_key,
        rationale=candidate.rationale,
        size_fraction=size_fraction,
        target_profit_pct=candidate.target_profit_pct,
        max_risk_pct=candidate.max_risk_pct,
        candidate=candidate,
        legs=legs,
        metadata={**sizing_meta, "size_model": size_model},
    )
