from __future__ import annotations

import math
from dataclasses import replace

from rlm.roee.risk import (
    is_tradeable_environment,
    should_require_defined_risk,
)
from rlm.roee.sizing import (
    apply_uncertainty_vault,
    compute_confidence,
    compute_regime_adjusted_kelly_fraction,
    compute_regime_penalty_multiplier,
    compute_size_fraction,
    kelly_confidence_from_uncertainty,
    kelly_voltarget_size,
    parse_latent_regime_label,
    quantize_fraction,
)
from rlm.roee.strategy_map import get_strategy_for_regime
from rlm.roee.strike_selection import build_legs_from_candidate
from rlm.types.options import TradeCandidate, TradeDecision

# Maximum number of overlay signals that may modify size on a single bar.
# More than 3 creates combinatorial overfitting risk.
MAX_OVERLAY_SIGNALS = 3


# ---------------------------------------------------------------------------
# Mode A — Core Engine
# Regime classification + confidence gating + base sizing only.
# No pattern overlays. This is the baseline truth model.
# ---------------------------------------------------------------------------


def resolve_strategy_name(
    *,
    direction_regime: str,
    volatility_regime: str,
    liquidity_regime: str,
    dealer_flow_regime: str,
    short_dte: bool,
) -> str:
    candidate = get_strategy_for_regime(
        direction=direction_regime,
        volatility=volatility_regime,
        liquidity=liquidity_regime,
        dealer_flow=dealer_flow_regime,
        short_dte=short_dte,
    )
    return candidate.strategy_name


def build_candidate_from_strategy_name(
    strategy_name: str,
    *,
    direction_regime: str,
    volatility_regime: str,
    liquidity_regime: str,
    dealer_flow_regime: str,
    short_dte: bool,
) -> TradeCandidate:
    regime_key = f"{direction_regime}|{volatility_regime}|{liquidity_regime}|{dealer_flow_regime}"

    if strategy_name == "bull_call_spread":
        return TradeCandidate(
            strategy_name="bull_call_spread",
            regime_key=regime_key,
            rationale="Coordinate router selected bullish debit call spread.",
            target_dte_min=14,
            target_dte_max=35,
            target_profit_pct=0.45,
            max_risk_pct=0.02,
            long_sigma=0.5,
            short_sigma=1.5,
            defined_risk=True,
        )
    if strategy_name == "bear_put_spread":
        return TradeCandidate(
            strategy_name="bear_put_spread",
            regime_key=regime_key,
            rationale="Coordinate router selected bearish debit put spread.",
            target_dte_min=14,
            target_dte_max=35,
            target_profit_pct=0.45,
            max_risk_pct=0.02,
            long_sigma=-0.5,
            short_sigma=-1.5,
            defined_risk=True,
        )
    if strategy_name == "iron_condor":
        return TradeCandidate(
            strategy_name="iron_condor",
            regime_key=regime_key,
            rationale="Coordinate router selected range-bound iron condor structure.",
            target_dte_min=20,
            target_dte_max=45,
            target_profit_pct=0.40,
            max_risk_pct=0.02,
            wings_sigma_low=1.0,
            wings_sigma_high=1.5,
            defined_risk=True,
        )
    if strategy_name == "debit_spread":
        option_type = "call" if direction_regime == "bull" else "put"
        long_sigma = 0.5 if option_type == "call" else -0.5
        short_sigma = 1.5 if option_type == "call" else -1.5
        return TradeCandidate(
            strategy_name=f"debit_spread_{option_type}",
            regime_key=regime_key,
            rationale="Coordinate router selected directional debit spread.",
            target_dte_min=14,
            target_dte_max=35,
            target_profit_pct=0.45,
            max_risk_pct=0.02,
            long_sigma=long_sigma,
            short_sigma=short_sigma,
            defined_risk=True,
        )
    if strategy_name == "calendar_spread":
        return TradeCandidate(
            strategy_name="calendar_spread",
            regime_key=regime_key,
            rationale="Coordinate router selected calendar spread for transition regime.",
            target_dte_min=20,
            target_dte_max=45,
            target_profit_pct=0.35,
            max_risk_pct=0.015,
            long_sigma=0.0,
            short_sigma=0.0,
            defined_risk=True,
        )
    if strategy_name == "no_trade":
        return TradeCandidate(
            strategy_name="no_trade_or_micro_position",
            regime_key=regime_key,
            rationale="Coordinate router selected no trade.",
            target_dte_min=14,
            target_dte_max=21,
            target_profit_pct=0.25,
            max_risk_pct=0.005,
            defined_risk=True,
        )

    legacy_candidate = get_strategy_for_regime(
        direction=direction_regime,
        volatility=volatility_regime,
        liquidity=liquidity_regime,
        dealer_flow=dealer_flow_regime,
        short_dte=short_dte,
    )
    if strategy_name == legacy_candidate.strategy_name:
        return legacy_candidate
    return replace(legacy_candidate, strategy_name=strategy_name)


def _core_trade_decision(
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
    strike_increment: float,
    short_dte: bool,
    forecast_return: float | None,
    forecast_uncertainty: float | None,
    realized_vol: float | None,
    use_dynamic_sizing: bool,
    vol_target: float,
    max_kelly_fraction: float,
    max_capital_fraction: float,
    regime_adjusted_kelly: bool,
    regime_state_label: str | None,
    regime_state_confidence: float | None,
    high_vol_kelly_multiplier: float,
    transition_kelly_multiplier: float,
    calm_trend_kelly_multiplier: float,
    vault_uncertainty_threshold: float | None,
    vault_size_multiplier: float,
) -> tuple[TradeDecision, dict]:
    """
    Core engine: regime + confidence + base sizing only.

    Returns the raw TradeDecision plus sizing metadata so the overlay engine
    can read and extend them without recomputing.
    """
    strategy_name = resolve_strategy_name(
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        short_dte=short_dte,
    )
    return _core_trade_decision_from_strategy_name(
        strategy_name=strategy_name,
        current_price=current_price,
        sigma=sigma,
        s_d=s_d,
        s_v=s_v,
        s_l=s_l,
        s_g=s_g,
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        regime_key=regime_key,
        strike_increment=strike_increment,
        short_dte=short_dte,
        forecast_return=forecast_return,
        forecast_uncertainty=forecast_uncertainty,
        realized_vol=realized_vol,
        use_dynamic_sizing=use_dynamic_sizing,
        vol_target=vol_target,
        max_kelly_fraction=max_kelly_fraction,
        max_capital_fraction=max_capital_fraction,
        regime_adjusted_kelly=regime_adjusted_kelly,
        regime_state_label=regime_state_label,
        regime_state_confidence=regime_state_confidence,
        high_vol_kelly_multiplier=high_vol_kelly_multiplier,
        transition_kelly_multiplier=transition_kelly_multiplier,
        calm_trend_kelly_multiplier=calm_trend_kelly_multiplier,
        vault_uncertainty_threshold=vault_uncertainty_threshold,
        vault_size_multiplier=vault_size_multiplier,
    )


def _core_trade_decision_from_strategy_name(
    *,
    strategy_name: str,
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
    strike_increment: float,
    short_dte: bool,
    forecast_return: float | None,
    forecast_uncertainty: float | None,
    realized_vol: float | None,
    use_dynamic_sizing: bool,
    vol_target: float,
    max_kelly_fraction: float,
    max_capital_fraction: float,
    regime_adjusted_kelly: bool,
    regime_state_label: str | None,
    regime_state_confidence: float | None,
    high_vol_kelly_multiplier: float,
    transition_kelly_multiplier: float,
    calm_trend_kelly_multiplier: float,
    vault_uncertainty_threshold: float | None,
    vault_size_multiplier: float,
) -> tuple[TradeDecision, dict]:
    candidate = build_candidate_from_strategy_name(
        strategy_name,
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        short_dte=short_dte,
    )

    if candidate.strategy_name == "no_trade_or_micro_position":
        return (
            TradeDecision(
                action="skip",
                strategy_name=candidate.strategy_name,
                regime_key=regime_key,
                rationale=candidate.rationale,
                candidate=candidate,
            ),
            {},
        )

    require_defined_risk = should_require_defined_risk(s_l, s_g)
    if require_defined_risk and not candidate.defined_risk:
        return (
            TradeDecision(
                action="skip",
                strategy_name=candidate.strategy_name,
                regime_key=regime_key,
                rationale="Candidate rejected: defined-risk required.",
                candidate=candidate,
            ),
            {},
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
        kelly_direction_regime, kelly_volatility_regime = parse_latent_regime_label(regime_state_label)
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
        kelly_confidence = kelly_confidence_from_uncertainty(forecast_uncertainty=forecast_uncertainty)
        size_fraction = quantize_fraction(
            min(
                candidate.max_risk_pct,
                max_capital_fraction,
                raw_dynamic_size * regime_penalty * kelly_confidence,
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
                "kelly_confidence_multiplier": kelly_confidence,
                "regime_adjusted_kelly": regime_adjusted_kelly,
                "regime_state_label": regime_state_label or "",
                "regime_state_confidence": (
                    float(regime_state_confidence)
                    if regime_state_confidence is not None and math.isfinite(float(regime_state_confidence))
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
    sizing_meta["size_model"] = size_model

    if size_fraction <= 0:
        return (
            TradeDecision(
                action="skip",
                strategy_name=candidate.strategy_name,
                regime_key=regime_key,
                rationale="Position size reduced to zero by risk controls.",
                candidate=candidate,
            ),
            {},
        )

    legs = build_legs_from_candidate(
        candidate=candidate,
        current_price=current_price,
        sigma=sigma,
        strike_increment=strike_increment,
    )

    decision = TradeDecision(
        action="enter",
        strategy_name=candidate.strategy_name,
        regime_key=regime_key,
        rationale=candidate.rationale,
        size_fraction=size_fraction,
        target_profit_pct=candidate.target_profit_pct,
        max_risk_pct=candidate.max_risk_pct,
        candidate=candidate,
        legs=legs,
        metadata=sizing_meta,
    )
    return decision, sizing_meta


# ---------------------------------------------------------------------------
# Mode B — Overlay Engine
# Post-signal modifiers applied AFTER the core engine produces a decision.
# NOT part of signal generation — purely adjusts size of an existing signal.
# Hard cap: MAX_OVERLAY_SIGNALS active modifiers per trade.
# ---------------------------------------------------------------------------


def apply_overlay_engine(
    decision: TradeDecision,
    *,
    direction_regime: str,
    regime_state_label: str | None = None,
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
    Overlay engine: applies post-signal size modifiers to a core-engine decision.

    Only called when decision.action == "enter". Signals are ranked by strength
    and capped at MAX_OVERLAY_SIGNALS to prevent combinatorial overfit.
    """
    if decision.action != "enter":
        return decision

    hmm_direction_regime, hmm_volatility_regime = parse_latent_regime_label(regime_state_label)

    mtf_score = float(mtf_confluence_score) if mtf_confluence_score is not None else 0.0
    sweep_score = (
        float(mtf_confluence_liquidity_sweep_confirmed) if mtf_confluence_liquidity_sweep_confirmed is not None else 0.0
    )
    pool_score = float(pool_confluence_score) if pool_confluence_score is not None else 0.0
    orderflow_score = float(orderflow_confluence_score) if orderflow_confluence_score is not None else 0.0
    fvg_score = float(fvg_alignment_score) if fvg_alignment_score is not None else 0.0
    ob_score = float(order_block_alignment_score) if order_block_alignment_score is not None else 0.0
    sr_score = float(support_resistance_alignment_score) if support_resistance_alignment_score is not None else 0.0
    bullish_candle = float(bullish_candle_pattern_score) if bullish_candle_pattern_score is not None else 0.0
    bearish_candle = float(bearish_candle_pattern_score) if bearish_candle_pattern_score is not None else 0.0

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
    hmm_state_allows_sweep_boost = (
        hmm_direction_regime == direction_regime and hmm_direction_regime in {"bull", "bear"}
    ) and hmm_volatility_regime not in {"high", "crisis", "panic"}
    orderflow_direction_aligned = direction_regime in {"bull", "bear"} and (
        hmm_direction_regime in {"", direction_regime} or hmm_direction_regime == direction_regime
    )

    # Collect candidate signals as (boost_multiplier, signal_name, guard).
    # Sort by boost magnitude and take at most MAX_OVERLAY_SIGNALS.
    candidates: list[tuple[float, str]] = []
    if mtf_score >= 1.5 and (fvg_score >= 1.0 or ob_score >= 1.0):
        candidates.append((1.1, "mtf_plus_fvg_or_ob"))
    if bullish_structure_confirmed or bearish_structure_confirmed:
        candidates.append((1.1, "liquidity_pool_candle_sr_alignment"))
    if sweep_score > 2.0 and pool_score >= 2.0 and hmm_state_allows_sweep_boost:
        candidates.append((1.25, "hmm_sweep_pool_boost"))
    if orderflow_score > 2.5 and orderflow_direction_aligned:
        candidates.append((1.15, "orderflow_confluence_directional_boost"))

    # Enforce hard cap: strongest signals win.
    candidates.sort(key=lambda x: x[0], reverse=True)
    active = candidates[:MAX_OVERLAY_SIGNALS]

    overlay_multiplier = 1.0
    overlay_signals: list[str] = []
    for boost, name in active:
        overlay_multiplier *= boost
        overlay_signals.append(name)

    max_risk_pct = float(decision.max_risk_pct or 0.5)
    size_fraction = float(decision.size_fraction or 0.0)
    if overlay_multiplier != 1.0:
        size_fraction = quantize_fraction(min(max_risk_pct, 0.5, size_fraction * overlay_multiplier))

    meta = dict(decision.metadata)
    meta.update(
        {
            "overlay_multiplier": overlay_multiplier,
            "overlay_signals": "|".join(overlay_signals),
            "overlay_signal_count": len(active),
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

    from dataclasses import replace

    return replace(decision, size_fraction=size_fraction, metadata=meta)


# ---------------------------------------------------------------------------
# Public entry point — thin coordinator
# ---------------------------------------------------------------------------


def select_trade_from_strategy_name(
    *,
    strategy_name: str,
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
    volume_ratio: float | None = None,
    regime_transition: bool = False,
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
    use_volume_profile_gating: bool = False,
    effort_result_divergence: float | None = None,
    auction_state: str | None = None,
    eighty_percent_rule_signal: bool = False,
    cumulative_wyckoff_score: float | None = None,
    wyckoff_threshold: float = 0.7,
    balance_haircut: float = 0.5,
    eighty_percent_boost: float = 0.2,
    hybrid_strength_scaling: bool = True,
    hybrid_strength: float | None = None,
    gex_confluence_enabled: bool = True,
    gex_confluence_poc: float | None = None,
    # Overlay signals — passed to Mode B only
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
    if not math.isfinite(current_price) or current_price <= 0:
        return TradeDecision(action="skip", rationale="Invalid current price.")

    if not math.isfinite(sigma) or sigma <= 0:
        return TradeDecision(action="skip", rationale="Invalid sigma.")

    env_ok, env_reason = is_tradeable_environment(
        has_major_event=has_major_event,
        bid_ask_spread_pct=bid_ask_spread_pct,
        volume_ratio=volume_ratio,
        regime_transition=regime_transition,
    )
    if not env_ok:
        return TradeDecision(action="skip", rationale=env_reason)

    decision, _ = _core_trade_decision_from_strategy_name(
        strategy_name=strategy_name,
        current_price=current_price,
        sigma=sigma,
        s_d=s_d,
        s_v=s_v,
        s_l=s_l,
        s_g=s_g,
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        regime_key=regime_key,
        strike_increment=strike_increment,
        short_dte=short_dte,
        forecast_return=forecast_return,
        forecast_uncertainty=forecast_uncertainty,
        realized_vol=realized_vol,
        use_dynamic_sizing=use_dynamic_sizing,
        vol_target=vol_target,
        max_kelly_fraction=max_kelly_fraction,
        max_capital_fraction=max_capital_fraction,
        regime_adjusted_kelly=regime_adjusted_kelly,
        regime_state_label=regime_state_label,
        regime_state_confidence=regime_state_confidence,
        high_vol_kelly_multiplier=high_vol_kelly_multiplier,
        transition_kelly_multiplier=transition_kelly_multiplier,
        calm_trend_kelly_multiplier=calm_trend_kelly_multiplier,
        vault_uncertainty_threshold=vault_uncertainty_threshold,
        vault_size_multiplier=vault_size_multiplier,
    )

    if use_volume_profile_gating and decision.action == "enter":
        vp_meta = dict(decision.metadata)
        divergence = (
            float(effort_result_divergence)
            if effort_result_divergence is not None and math.isfinite(float(effort_result_divergence))
            else 0.0
        )

        forecast_sign = 0
        if forecast_return is not None and math.isfinite(float(forecast_return)):
            forecast_sign = 1 if float(forecast_return) > 0 else (-1 if float(forecast_return) < 0 else 0)
        elif direction_regime == "bull":
            forecast_sign = 1
        elif direction_regime == "bear":
            forecast_sign = -1

        wyckoff_score = (
            float(cumulative_wyckoff_score)
            if cumulative_wyckoff_score is not None and math.isfinite(float(cumulative_wyckoff_score))
            else abs(divergence)
        )
        if wyckoff_score >= float(wyckoff_threshold) and (
            (forecast_sign > 0 and divergence < 0) or (forecast_sign < 0 and divergence > 0)
        ):
            return TradeDecision(
                action="skip",
                strategy_name=decision.strategy_name,
                regime_key=decision.regime_key,
                rationale="Volume-profile gate: absorption opposes forecast direction.",
                candidate=decision.candidate,
                metadata={**vp_meta, "vp_gated": True, "vp_gate_reason": "absorption_conflict"},
            )

        size_fraction = float(decision.size_fraction or 0.0)
        if str(auction_state or "").lower() == "balance" and forecast_sign != 0:
            size_fraction = quantize_fraction(size_fraction * float(balance_haircut))
            vp_meta["vp_balance_size_reduction"] = float(balance_haircut)

        confidence_boost = float(eighty_percent_boost) if eighty_percent_rule_signal else 0.0
        vp_meta["vp_80_percent_rule_signal"] = bool(eighty_percent_rule_signal)
        vp_meta["vp_confidence_boost"] = confidence_boost
        vp_meta["vp_gated"] = False

        if hybrid_strength_scaling and hybrid_strength is not None and math.isfinite(float(hybrid_strength)):
            hs = float(min(max(float(hybrid_strength), 0.5), 1.5))
            size_fraction = quantize_fraction(size_fraction * hs)
            vp_meta["vp_hybrid_strength_multiplier"] = hs

        if gex_confluence_enabled and gex_confluence_poc is not None and math.isfinite(float(gex_confluence_poc)):
            gex_sign = 1 if float(gex_confluence_poc) > 0 else (-1 if float(gex_confluence_poc) < 0 else 0)
            trade_sign = forecast_sign
            aligned = (gex_sign > 0 and trade_sign == 0) or (gex_sign < 0 and trade_sign != 0)
            vp_meta["vp_gex_confluence_poc"] = float(gex_confluence_poc)
            vp_meta["vp_gex_style"] = "mean_reversion" if gex_sign > 0 else ("momentum" if gex_sign < 0 else "neutral")
            if not aligned and gex_sign != 0:
                size_fraction = quantize_fraction(size_fraction * 0.75)
                vp_meta["vp_gex_misalignment_haircut"] = 0.75

        if confidence_boost > 0:
            base_conf = float(vp_meta.get("confidence", 0.0) or 0.0)
            vp_meta["confidence"] = float(min(1.0, base_conf + confidence_boost))

        decision = replace(decision, size_fraction=size_fraction, metadata=vp_meta)

    decision = apply_overlay_engine(
        decision,
        direction_regime=direction_regime,
        regime_state_label=regime_state_label,
        mtf_confluence_score=mtf_confluence_score,
        mtf_confluence_liquidity_sweep_confirmed=mtf_confluence_liquidity_sweep_confirmed,
        pool_confluence_score=pool_confluence_score,
        orderflow_confluence_score=orderflow_confluence_score,
        bullish_liquidity_pool_nearby=bullish_liquidity_pool_nearby,
        bearish_liquidity_pool_nearby=bearish_liquidity_pool_nearby,
        fvg_alignment_score=fvg_alignment_score,
        order_block_alignment_score=order_block_alignment_score,
        bullish_candle_pattern_score=bullish_candle_pattern_score,
        bearish_candle_pattern_score=bearish_candle_pattern_score,
        support_resistance_alignment_score=support_resistance_alignment_score,
    )

    return decision


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
    volume_ratio: float | None = None,
    regime_transition: bool = False,
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
    use_volume_profile_gating: bool = False,
    effort_result_divergence: float | None = None,
    auction_state: str | None = None,
    eighty_percent_rule_signal: bool = False,
    cumulative_wyckoff_score: float | None = None,
    wyckoff_threshold: float = 0.7,
    balance_haircut: float = 0.5,
    eighty_percent_boost: float = 0.2,
    hybrid_strength_scaling: bool = True,
    hybrid_strength: float | None = None,
    gex_confluence_enabled: bool = True,
    gex_confluence_poc: float | None = None,
    # Overlay signals — passed to Mode B only
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
    Main ROEE entry point for one bar.

    Delegates to the core engine (Mode A) then the overlay engine (Mode B).
    Environment gates are checked before either engine runs.

    Parameters
    ----------
    short_dte:
        Activates 0DTE / 1DTE intraday strategy selection.
    volume_ratio:
        Current bar volume / rolling average. Values < 0.5 trigger no-trade.
    regime_transition:
        True when the regime model flags an active state transition.
    """
    strategy_name = resolve_strategy_name(
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        short_dte=short_dte,
    )
    return select_trade_from_strategy_name(
        strategy_name=strategy_name,
        current_price=current_price,
        sigma=sigma,
        s_d=s_d,
        s_v=s_v,
        s_l=s_l,
        s_g=s_g,
        direction_regime=direction_regime,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        regime_key=regime_key,
        bid_ask_spread_pct=bid_ask_spread_pct,
        has_major_event=has_major_event,
        volume_ratio=volume_ratio,
        regime_transition=regime_transition,
        strike_increment=strike_increment,
        short_dte=short_dte,
        forecast_return=forecast_return,
        forecast_uncertainty=forecast_uncertainty,
        realized_vol=realized_vol,
        use_dynamic_sizing=use_dynamic_sizing,
        vol_target=vol_target,
        max_kelly_fraction=max_kelly_fraction,
        max_capital_fraction=max_capital_fraction,
        regime_adjusted_kelly=regime_adjusted_kelly,
        regime_state_label=regime_state_label,
        regime_state_confidence=regime_state_confidence,
        high_vol_kelly_multiplier=high_vol_kelly_multiplier,
        transition_kelly_multiplier=transition_kelly_multiplier,
        calm_trend_kelly_multiplier=calm_trend_kelly_multiplier,
        vault_uncertainty_threshold=vault_uncertainty_threshold,
        vault_size_multiplier=vault_size_multiplier,
        use_volume_profile_gating=use_volume_profile_gating,
        effort_result_divergence=effort_result_divergence,
        auction_state=auction_state,
        eighty_percent_rule_signal=eighty_percent_rule_signal,
        cumulative_wyckoff_score=cumulative_wyckoff_score,
        wyckoff_threshold=wyckoff_threshold,
        balance_haircut=balance_haircut,
        eighty_percent_boost=eighty_percent_boost,
        hybrid_strength_scaling=hybrid_strength_scaling,
        hybrid_strength=hybrid_strength,
        gex_confluence_enabled=gex_confluence_enabled,
        gex_confluence_poc=gex_confluence_poc,
        mtf_confluence_score=mtf_confluence_score,
        mtf_confluence_liquidity_sweep_confirmed=mtf_confluence_liquidity_sweep_confirmed,
        pool_confluence_score=pool_confluence_score,
        orderflow_confluence_score=orderflow_confluence_score,
        bullish_liquidity_pool_nearby=bullish_liquidity_pool_nearby,
        bearish_liquidity_pool_nearby=bearish_liquidity_pool_nearby,
        fvg_alignment_score=fvg_alignment_score,
        order_block_alignment_score=order_block_alignment_score,
        bullish_candle_pattern_score=bullish_candle_pattern_score,
        bearish_candle_pattern_score=bearish_candle_pattern_score,
        support_resistance_alignment_score=support_resistance_alignment_score,
    )
