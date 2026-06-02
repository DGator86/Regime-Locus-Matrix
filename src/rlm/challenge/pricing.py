"""Simplified option pricing utilities for the dry-run challenge simulator.

No external dependencies.  Uses Bachelier-style ATM approximation and a
first-order Greeks update for P&L simulation between sessions.
"""

from __future__ import annotations

import math

# Imported lazily to avoid circular dependency — used only by type hints below.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.challenge.config import ChallengeConfig

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

    Uses delta P&L + gamma convexity + sqrt-time accelerated theta.  Theta is
    proportional to 1/√t so decay is slow early in the option's life and
    accelerates near expiry, matching the shape of real options theta.  The
    normalization ensures total decay from entry to expiry integrates to
    ``entry_premium`` (i.e. the option fully decays if the underlying doesn't move).
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

    # Theta with sqrt-time acceleration: theta ∝ 1/√t so decay accelerates near expiry.
    # Normalization: integral of 1/√t from 0 to T is 2√T, so total decay = entry_premium.
    # theta_pnl = -entry_premium × (√t_prev − √t_now) / √t_entry
    dte_at_entry = dte_remaining + days_elapsed
    t_entry = max(dte_at_entry, 1) / 252.0
    t_prev = max(dte_remaining + days_elapsed, 1) / 252.0
    # sqrt(0) = 0 is well-defined; no floor needed.  When dte_remaining=0 the
    # full remaining premium correctly decays to zero (intrinsic value aside).
    t_now = max(dte_remaining, 0) / 252.0
    theta_pnl = -entry_premium * (math.sqrt(t_prev) - math.sqrt(t_now)) / math.sqrt(t_entry)

    new_price = entry_premium + delta_pnl + gamma_pnl + theta_pnl
    return max(new_price, 0.01)


# ---------------------------------------------------------------------------
# Friction model (spread + commission)
# ---------------------------------------------------------------------------


def min_tick_round(premium: float) -> float:
    """Round option premium to nearest penny (standard US equity option increment)."""
    return round(premium * 100.0) / 100.0


def spread_cost(premium: float, qty: int, cfg: "ChallengeConfig") -> float:
    """Total round-trip friction for a position: spread (entry+exit) + 2× commissions.

    Parameters
    ----------
    premium:
        Per-share option price at entry.
    qty:
        Number of contracts (each = 100 shares).
    cfg:
        ChallengeConfig carrying spread_half_width_frac and commission_per_contract.

    Returns
    -------
    float
        Total friction in dollars applied across the full round trip.
    """
    if not cfg.use_spread_model:
        return 0.0
    # Entry half-spread + exit half-spread = full spread × notional
    spread_dollars = 2.0 * cfg.spread_half_width_frac * premium * qty * 100
    commission_dollars = 2.0 * cfg.commission_per_contract * qty  # entry leg + exit leg
    return spread_dollars + commission_dollars


def entry_friction(premium: float, qty: int, cfg: "ChallengeConfig") -> float:
    """One-way friction charged at entry: half-spread + 1× commission per contract."""
    if not cfg.use_spread_model:
        return 0.0
    spread_dollars = cfg.spread_half_width_frac * premium * qty * 100
    commission_dollars = cfg.commission_per_contract * qty
    return spread_dollars + commission_dollars


def exit_friction(fill_premium: float, qty: int, cfg: "ChallengeConfig") -> float:
    """One-way friction charged at exit: half-spread + 1× commission per contract."""
    if not cfg.use_spread_model:
        return 0.0
    spread_dollars = cfg.spread_half_width_frac * fill_premium * qty * 100
    commission_dollars = cfg.commission_per_contract * qty
    return spread_dollars + commission_dollars


# ---------------------------------------------------------------------------
# Delta/theta ratio for surface-aware strike selection (Item 8)
# ---------------------------------------------------------------------------


def delta_per_theta_ratio(
    underlying: float,
    strike: float,
    iv: float,
    dte: int,
    option_type: str,
) -> float:
    """Ratio of |delta| to daily theta cost — higher means more directional bang per decay dollar.

    Parameters
    ----------
    underlying:
        Current price of the underlying.
    strike:
        Option strike price.
    iv:
        Implied volatility (annualised, e.g. 0.20).
    dte:
        Days to expiry.
    option_type:
        ``"call"`` or ``"put"``.

    Returns
    -------
    float
        |delta| / daily_theta, or 0.0 if theta is negligible.
    """
    d = abs(estimate_delta(underlying, strike, iv, dte, option_type))
    premium = estimate_premium(underlying, iv, dte, strike)
    t_prev = max(dte, 1) / 252.0
    t_now = max(dte - 1, 0.5) / 252.0
    t_entry = t_prev  # normalise relative to entry life
    daily_theta = premium * (math.sqrt(t_prev) - math.sqrt(t_now)) / math.sqrt(t_entry)
    if daily_theta <= 1e-9:
        return 0.0
    return d / daily_theta
