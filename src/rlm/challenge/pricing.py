"""Simplified option pricing utilities for the dry-run challenge simulator.

No external dependencies.  Uses Bachelier-style ATM approximation and a
first-order Greeks update for P&L simulation between sessions.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Premium estimation
# ---------------------------------------------------------------------------


def atm_premium(underlying: float, iv: float, dte: int) -> float:
    """Estimate ATM option premium via the Bachelier approximation.

    Formula: S · σ · √T · N'(0) ≈ S · σ · √(dte/252) · 0.3989
    """
    t = max(dte, 1) / 252.0
    return underlying * iv * math.sqrt(t) * 0.3989


def otm_premium(underlying: float, iv: float, dte: int, strike: float) -> float:
    """Estimate OTM option premium with a lognormal moneyness discount."""
    base = atm_premium(underlying, iv, dte)
    t = max(dte, 1) / 252.0
    moneyness = abs(math.log(strike / underlying)) / (iv * math.sqrt(t) + 1e-9)
    # Vega-weighted discount: exponential falloff with moneyness in σ units
    discount = math.exp(-0.5 * moneyness**2)
    return max(base * discount, 0.01)


def estimate_premium(underlying: float, iv: float, dte: int, strike: float) -> float:
    """Pick the right estimator depending on whether the option is ATM or OTM."""
    if abs(strike - underlying) / underlying < 0.001:
        return atm_premium(underlying, iv, dte)
    return otm_premium(underlying, iv, dte, strike)


# ---------------------------------------------------------------------------
# Delta approximation
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def estimate_delta(
    underlying: float,
    strike: float,
    iv: float,
    dte: int,
    option_type: str,
) -> float:
    """First-order delta via Black-Scholes d1 (simplified, no risk-free rate)."""
    t = max(dte, 1) / 252.0
    sigma_sqrt_t = iv * math.sqrt(t) + 1e-9
    d1 = math.log(underlying / strike) / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


# ---------------------------------------------------------------------------
# Position value update
# ---------------------------------------------------------------------------


def updated_premium(
    entry_premium: float,
    delta: float,
    underlying_entry: float,
    underlying_now: float,
    days_elapsed: int,
    dte_remaining: int,
    iv: float,
) -> float:
    """Approximate new per-share option premium after an underlying move + time decay.

    Uses delta P&L + gamma convexity + linear theta approximation.
    """
    move = underlying_now - underlying_entry

    # Delta contribution
    delta_pnl = delta * move

    # Gamma (rough): Γ ≈ N'(d1) / (S·σ·√T)
    t = max(dte_remaining, 1) / 252.0
    sigma_sqrt_t = iv * math.sqrt(t) + 1e-9
    d1 = math.log(underlying_now / (underlying_entry + 1e-9)) / sigma_sqrt_t + 0.5 * sigma_sqrt_t
    gamma = math.exp(-0.5 * d1**2) / (math.sqrt(2 * math.pi) * underlying_now * sigma_sqrt_t)
    gamma_pnl = 0.5 * gamma * move**2

    # Theta: −entry_premium / (dte_at_entry × 1.4)  per day
    dte_at_entry = dte_remaining + days_elapsed
    theta_per_day = -entry_premium / (max(dte_at_entry, 1) * 1.4)
    theta_pnl = theta_per_day * days_elapsed

    new_price = entry_premium + delta_pnl + gamma_pnl + theta_pnl
    return max(new_price, 0.01)
