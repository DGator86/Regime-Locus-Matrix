"""
Full Black-Scholes Greek suite for the RLM microstructure layer.

Extends ``rlm.data.bs_greeks`` (which provides delta, gamma, vega, vanna, charm)
with the complete second- and third-order Greek surface:

  First-order  : delta, gamma, theta, vega, rho
  Second-order : vanna, charm, vomma, veta
  Third-order  : speed, zomma, color, ultima

Also exposes a vectorised implied-volatility solver (Newton-Raphson + bisection
fallback) that works on raw bid/ask mid prices from an option chain DataFrame.

All formulae follow the standard Black-Scholes conventions; vega is expressed
*per unit of IV* (i.e. sigma=0.20 ⟹ 1% shift = 0.01 units).  To convert to
the "per 1 vega point" convention divide vega by 100.

Usage::

    from rlm.data.microstructure.calculators.greeks import full_greeks_row, solve_iv
    g = full_greeks_row(spot=440, strike=440, time_years=30/365, iv=0.18, is_call=True)
    iv = solve_iv(market_price=2.50, spot=440, strike=445, time_years=30/365, is_call=True)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import norm

_LOG2PI_HALF = 0.5 * math.log(2 * math.pi)  # pre-computed constant


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class GreekBundle:
    """All BS Greeks for one option contract, plus raw inputs."""

    # Inputs
    spot: float
    strike: float
    time_years: float
    iv: float
    risk_free: float
    is_call: bool

    # First-order
    delta: float = float("nan")
    gamma: float = float("nan")
    theta: float = float("nan")   # per calendar day (not per year)
    vega: float = float("nan")    # per 1-unit IV shift
    rho: float = float("nan")

    # Second-order
    vanna: float = float("nan")
    charm: float = float("nan")   # per calendar day
    vomma: float = float("nan")
    veta: float = float("nan")    # vega decay per calendar day

    # Third-order
    speed: float = float("nan")
    zomma: float = float("nan")
    color: float = float("nan")   # gamma decay per calendar day
    ultima: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta, "gamma": self.gamma,
            "theta": self.theta, "vega": self.vega, "rho": self.rho,
            "vanna": self.vanna, "charm": self.charm,
            "vomma": self.vomma, "veta": self.veta,
            "speed": self.speed, "zomma": self.zomma,
            "color": self.color, "ultima": self.ultima,
        }


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def full_greeks_row(
    *,
    spot: float,
    strike: float,
    time_years: float,
    iv: float,
    risk_free: float = 0.052,
    is_call: bool = True,
) -> GreekBundle:
    """
    Compute all 13 Black-Scholes Greeks for a single option contract.

    Parameters
    ----------
    spot        : Current underlying price (S)
    strike      : Option strike price (K)
    time_years  : Time to expiry in years (T)
    iv          : Implied volatility as a decimal (e.g. 0.20 = 20%)
    risk_free   : Continuously-compounded risk-free rate (default 5.2%)
    is_call     : True for call, False for put

    Returns
    -------
    GreekBundle with all fields populated.  Returns NaN fields if inputs are
    degenerate (T ≤ 0, iv ≤ 0, spot ≤ 0, strike ≤ 0).
    """
    S, K, T, sig, r = float(spot), float(strike), float(time_years), float(iv), float(risk_free)
    bundle = GreekBundle(spot=S, strike=K, time_years=T, iv=sig, risk_free=r, is_call=is_call)

    if S <= 0.0 or K <= 0.0 or T < 1e-10 or sig < 1e-10:
        return bundle

    sqrt_T = math.sqrt(T)
    sig_sqrt_T = sig * sqrt_T
    disc = math.exp(-r * T)

    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T

    phi_d1 = float(norm.pdf(d1))   # N'(d1)
    Phi_d1 = float(norm.cdf(d1))   # N(d1)
    Phi_d2 = float(norm.cdf(d2))   # N(d2)

    # ── First order ──────────────────────────────────────────────────────────
    bundle.delta = Phi_d1 if is_call else (Phi_d1 - 1.0)
    bundle.gamma = phi_d1 / (S * sig_sqrt_T)
    bundle.vega = S * phi_d1 * sqrt_T                      # per 1.0 unit IV

    # Theta: per calendar day (divide annual by 365)
    common_theta = -S * phi_d1 * sig / (2.0 * sqrt_T)
    if is_call:
        bundle.theta = (common_theta - r * K * disc * Phi_d2) / 365.0
    else:
        bundle.theta = (common_theta + r * K * disc * (1.0 - Phi_d2)) / 365.0

    # Rho: per 1.0 unit interest rate (i.e. 100 basis point move)
    if is_call:
        bundle.rho = K * T * disc * Phi_d2
    else:
        bundle.rho = -K * T * disc * (1.0 - Phi_d2)

    # ── Second order ─────────────────────────────────────────────────────────
    # Vanna: ∂delta/∂σ  = ∂vega/∂S  = -φ(d1)·d2/σ
    bundle.vanna = -phi_d1 * d2 / sig if sig > 1e-12 else 0.0

    # Charm: ∂delta/∂T per calendar day
    if T > 1e-10 and sig_sqrt_T > 1e-12:
        bundle.charm = (
            -phi_d1 * (2.0 * r * T - d2 * sig_sqrt_T) / (2.0 * T * sig_sqrt_T)
        ) / 365.0

    # Vomma: ∂vega/∂σ  (also known as volga)
    if sig > 1e-12:
        bundle.vomma = bundle.vega * d1 * d2 / sig

    # Veta: ∂vega/∂T per calendar day
    if T > 1e-10 and sig_sqrt_T > 1e-12:
        bundle.veta = (
            -S * phi_d1 * sqrt_T
            * (r * d1 / sig_sqrt_T - (1.0 + d1 * d2) / (2.0 * T))
        ) / 365.0

    # ── Third order ──────────────────────────────────────────────────────────
    # Speed: ∂gamma/∂S
    if sig_sqrt_T > 1e-12:
        bundle.speed = -bundle.gamma / S * (1.0 + d1 / sig_sqrt_T)

    # Zomma: ∂gamma/∂σ
    if sig > 1e-12:
        bundle.zomma = bundle.gamma * (d1 * d2 - 1.0) / sig

    # Color: ∂gamma/∂T per calendar day
    if T > 1e-10 and sig_sqrt_T > 1e-12:
        bundle.color = (
            -phi_d1
            / (2.0 * S * T * sig_sqrt_T)
            * (2.0 * r * T + 1.0 - d1 * (sig_sqrt_T - (2.0 * r * T * d2 / sig_sqrt_T)))
        ) / 365.0

    # Ultima: ∂vomma/∂σ
    if sig > 1e-12:
        bundle.ultima = (
            -bundle.vega
            / (sig * sig)
            * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)
        )

    return bundle


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float, sig: float, is_call: bool) -> float:
    """Black-Scholes option price."""
    if sig < 1e-10 or T < 1e-10:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt_T)
    d2 = d1 - sig * sqrt_T
    disc = math.exp(-r * T)

    if is_call:
        return S * float(norm.cdf(d1)) - K * disc * float(norm.cdf(d2))
    return K * disc * float(norm.cdf(-d2)) - S * float(norm.cdf(-d1))


def solve_iv(
    market_price: float,
    spot: float,
    strike: float,
    time_years: float,
    is_call: bool,
    risk_free: float = 0.052,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Implied volatility via Newton-Raphson (falls back to bisection).

    Returns NaN if the market_price is below intrinsic value or if the
    solver does not converge within *max_iter* iterations.
    """
    S, K, T, r = float(spot), float(strike), float(time_years), float(risk_free)

    if T < 1e-10 or market_price <= 0.0 or S <= 0.0 or K <= 0.0:
        return float("nan")

    disc = math.exp(-r * T)
    intrinsic = max(S - K * disc, 0.0) if is_call else max(K * disc - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return float("nan")

    # Seed from Brenner-Subrahmanyam approximation
    sqrt_T = math.sqrt(T)
    sig = math.sqrt(2.0 * math.pi / T) * market_price / S

    # Newton-Raphson
    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sig, is_call)
        vega_ = S * float(norm.pdf(
            (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrt_T)
        )) * sqrt_T
        if abs(vega_) < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return max(sig, 1e-6)
        sig = sig - diff / vega_
        if sig <= 0.0:
            sig = 1e-4

    # Bisection fallback
    lo, hi = 1e-4, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = _bs_price(S, K, T, r, mid, is_call)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    result = 0.5 * (lo + hi)
    return result if result > 1e-5 else float("nan")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def compute_greeks_dataframe(
    chain: "pd.DataFrame",
    risk_free: float = 0.052,
) -> "pd.DataFrame":
    """
    Vectorised Greek computation for an option chain DataFrame.

    Expected input columns: ``strike``, ``dte`` (days to expiry), ``mid`` or
    ``implied_vol``, ``option_type`` ('call'/'put'), ``underlying_price``.

    If ``implied_vol`` is not present, it is solved from ``mid``.

    Returns the input DataFrame with all Greek columns appended.
    """
    import pandas as pd

    out = chain.copy()

    # Ensure IV
    if "implied_vol" not in out.columns or out["implied_vol"].isna().all():
        if "mid" not in out.columns:
            raise ValueError("chain must have 'implied_vol' or 'mid' column")

        def _iv(row: "pd.Series") -> float:
            dte = float(row.get("dte", 0) or 0)
            T = dte / 365.0
            return solve_iv(
                market_price=float(row["mid"] or 0),
                spot=float(row["underlying_price"]),
                strike=float(row["strike"]),
                time_years=T,
                is_call=(str(row["option_type"]).lower() == "call"),
                risk_free=risk_free,
            )

        out["implied_vol"] = out.apply(_iv, axis=1)

    greek_cols: list[str] = [
        "delta", "gamma", "theta", "vega", "rho",
        "vanna", "charm", "vomma", "veta",
        "speed", "zomma", "color", "ultima",
    ]

    def _row_greeks(row: "pd.Series") -> "pd.Series":
        dte = float(row.get("dte", 0) or 0)
        T = max(dte / 365.0, 1e-10)
        g = full_greeks_row(
            spot=float(row["underlying_price"]),
            strike=float(row["strike"]),
            time_years=T,
            iv=float(row["implied_vol"] or 0),
            risk_free=risk_free,
            is_call=(str(row["option_type"]).lower() == "call"),
        )
        return pd.Series(g.as_dict())

    greeks_df = out.apply(_row_greeks, axis=1)
    for col in greek_cols:
        out[col] = greeks_df[col]

    return out
