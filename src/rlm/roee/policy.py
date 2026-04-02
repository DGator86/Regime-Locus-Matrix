from __future__ import annotations

import math

from rlm.roee.risk import (
    should_require_defined_risk,
    should_skip_for_event_risk,
    spread_quality_ok,
)
from rlm.roee.sizing import compute_confidence, compute_size_fraction
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
) -> TradeDecision:
    """
    Main ROEE entry point for one bar / one underlying snapshot.
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
        metadata={
            "confidence": confidence,
            "require_defined_risk": require_defined_risk,
            "current_price": current_price,
            "sigma": sigma,
        },
    )
