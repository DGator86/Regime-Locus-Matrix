"""Black–Scholes greeks (scipy) for synthetic chains and gap-filling."""

from __future__ import annotations

import math

from scipy.stats import norm


def bs_greeks_row(
    *,
    spot: float,
    strike: float,
    time_years: float,
    iv: float,
    risk_free: float = 0.052,
    is_call: bool = True,
) -> tuple[float, float, float, float, float]:
    """
    Returns (delta, gamma, vega, vanna, charm) per share.
    Vega is per 1.0 move in sigma (not 1% vol point).
    """
    if spot <= 0 or strike <= 0 or time_years <= 1e-10 or iv <= 1e-10:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    s, k, t, sig, r = float(spot), float(strike), float(time_years), float(iv), float(risk_free)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t
    pdf = float(norm.pdf(d1))

    gamma = pdf / (s * sig * sqrt_t)
    vega = s * pdf * sqrt_t

    if is_call:
        delta = float(norm.cdf(d1))
    else:
        delta = float(norm.cdf(d1)) - 1.0

    # Vanna: d(delta)/d(vol) scaled to sigma
    vanna = (vega / s) * (1.0 - d1 / (sig * sqrt_t)) if sig * sqrt_t > 1e-12 else 0.0

    # Charm (delta decay): d(delta)/dT — standard BS form
    charm = -pdf * (2.0 * r * t - d2 * sig * sqrt_t) / (2.0 * t * sig * sqrt_t)
    if t <= 1e-10:
        charm = math.nan

    return delta, gamma, vega, vanna, charm
