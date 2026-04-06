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
