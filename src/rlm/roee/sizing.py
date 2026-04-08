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


def parse_latent_regime_label(state_label: str | None) -> tuple[str | None, str | None]:
    if not state_label:
        return None, None

    normalized = str(state_label).removesuffix("_like")
    parts = [part.strip() for part in normalized.split("|")]
    direction = parts[0] if len(parts) >= 1 and parts[0] else None
    volatility = parts[1] if len(parts) >= 2 and parts[1] else None
    return direction, volatility


def compute_regime_adjusted_kelly_fraction(
    *,
    base_kelly_fraction: float,
    regime_state_label: str | None = None,
    regime_state_confidence: float | None = None,
    high_vol_multiplier: float = 0.5,
    transition_multiplier: float = 0.75,
    calm_trend_multiplier: float = 1.25,
) -> float:
    """
    Adjust the configured Kelly fraction using the latent regime label/confidence.

    High-vol regimes cut sizing, transition regimes de-risk moderately, and calm
    trending regimes earn a small boost.  Confidence blends the multiplier back
    toward neutral sizing when the latent regime assignment is uncertain.
    """
    base_fraction = quantize_fraction(clamp(base_kelly_fraction, 0.0, 1.0))
    direction_regime, volatility_regime = parse_latent_regime_label(regime_state_label)
    if direction_regime is None and volatility_regime is None:
        return base_fraction

    confidence = clamp(
        float(regime_state_confidence) if regime_state_confidence is not None else 1.0,
        0.0,
        1.0,
    )
    target_multiplier = 1.0
    if volatility_regime == "high_vol":
        target_multiplier = high_vol_multiplier
    elif direction_regime == "transition" or volatility_regime == "transition":
        target_multiplier = transition_multiplier
    elif volatility_regime == "low_vol" and direction_regime in {"bull", "bear"}:
        target_multiplier = calm_trend_multiplier

    blended_multiplier = 1.0 + ((target_multiplier - 1.0) * confidence)
    return quantize_fraction(clamp(base_fraction * blended_multiplier, 0.0, 1.0))


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

    kelly = forecast_return / (realized_vol**2)
    capped_kelly = min(kelly, max_kelly_fraction)
    size = (vol_target / realized_vol) * capped_kelly
    return quantize_fraction(clamp(size, 0.0, max_capital_fraction))


def apply_uncertainty_vault(
    *,
    size_fraction: float,
    forecast_uncertainty: float | None,
    uncertainty_threshold: float | None = 0.03,
    size_multiplier: float = 0.5,
) -> tuple[float, dict[str, float | bool]]:
    """
    Reduce risk when the forecast interval is wide enough to signal model confusion.
    """
    threshold_ok = uncertainty_threshold is not None and float(uncertainty_threshold) > 0.0
    multiplier = clamp(float(size_multiplier), 0.0, 1.0)
    uncertainty_ok = forecast_uncertainty is not None
    uncertainty_value = float(forecast_uncertainty) if uncertainty_ok else None
    uncertainty_finite = uncertainty_value is not None and uncertainty_value >= 0.0
    triggered = bool(
        threshold_ok and uncertainty_finite and uncertainty_value > float(uncertainty_threshold)
    )
    adjusted = size_fraction * multiplier if triggered else size_fraction
    metadata: dict[str, float | bool] = {
        "vault_enabled": bool(threshold_ok),
        "vault_triggered": triggered,
        "vault_size_multiplier": quantize_fraction(multiplier),
    }
    if threshold_ok:
        metadata["vault_uncertainty_threshold"] = float(uncertainty_threshold)
    if uncertainty_finite and uncertainty_value is not None:
        metadata["forecast_uncertainty"] = uncertainty_value
    return quantize_fraction(adjusted), metadata
