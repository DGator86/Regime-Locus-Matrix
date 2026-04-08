from __future__ import annotations


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def quantize_fraction(x: float, decimals: int = 10) -> float:
    """Stable percentages for TradeDecision / logs (avoids float noise on risk fractions)."""
    return float(round(x, decimals))


def compute_confidence(
    s_d: float,
    s_v: float,
    s_l: float,
    s_g: float,
) -> float:
    """
    Conservative first-pass confidence.
    Emphasize direction and liquidity, discount ambiguous states.
    Output in [0, 1].
    """
    base = 0.40 * abs(s_d) + 0.20 * abs(s_v) + 0.20 * abs(s_l) + 0.20 * abs(s_g)
    return quantize_fraction(clamp(base, 0.0, 1.0))


def compute_size_fraction(
    confidence: float,
    base_risk_pct: float,
    liquidity_regime: str,
    dealer_flow_regime: str,
    direction_regime: str,
) -> float:
    """
    Returns fraction of account risk budget to allocate.
    """
    size = base_risk_pct * confidence

    if liquidity_regime == "low_liquidity":
        size *= 0.6

    if dealer_flow_regime == "destabilizing":
        size *= 0.75

    if direction_regime == "transition":
        size *= 0.35

    return quantize_fraction(clamp(size, 0.0, base_risk_pct))


def compute_regime_penalty_multiplier(
    confidence: float,
    base_risk_pct: float,
    liquidity_regime: str,
    dealer_flow_regime: str,
    direction_regime: str,
) -> float:
    """
    Normalized penalty multiplier implied by the legacy size model.
    """
    if base_risk_pct <= 0:
        return 0.0
    sized = compute_size_fraction(
        confidence=confidence,
        base_risk_pct=base_risk_pct,
        liquidity_regime=liquidity_regime,
        dealer_flow_regime=dealer_flow_regime,
        direction_regime=direction_regime,
    )
    return quantize_fraction(clamp(sized / base_risk_pct, 0.0, 1.0))


def kelly_voltarget_size(
    *,
    forecast_return: float,
    realized_vol: float,
    vol_target: float = 0.15,
    max_kelly_fraction: float = 0.25,
    max_capital_fraction: float = 0.5,
) -> float:
    """
    Kelly-style size scaled to a target volatility.
    """
    if forecast_return <= 0.0 or realized_vol <= 1e-9:
        return 0.0

    kelly = forecast_return / (realized_vol ** 2)
    capped_kelly = min(kelly, max_kelly_fraction)
    size = (vol_target / realized_vol) * capped_kelly
    return quantize_fraction(clamp(size, 0.0, max_capital_fraction))
