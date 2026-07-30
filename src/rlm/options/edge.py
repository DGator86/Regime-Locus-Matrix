"""Options fair-value, Greek enrichment, and mispricing gates (universe / SPY short-DTE)."""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from rlm.data.microstructure.calculators.greeks import full_greeks_row, solve_iv

_DEFAULT_RISK_FREE = 0.052


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _years_to_expiry(expiry: str, *, as_of: date | None = None) -> float:
    try:
        exp = date.fromisoformat(str(expiry)[:10])
    except (TypeError, ValueError):
        return math.nan
    ref = as_of or datetime.now(timezone.utc).date()
    days = (exp - ref).days
    if days < 0:
        return 0.0
    return max(days / 365.0, 1.0 / (365.0 * 24.0))  # same-day ≈ 1 hour


def enrich_leg_greeks(
    leg: dict[str, Any],
    *,
    spot: float,
    risk_free: float = _DEFAULT_RISK_FREE,
) -> dict[str, Any]:
    """Fill missing delta/gamma/theta/vega/iv on a matched leg dict.

    When IV is solved from the leg mid, sets ``_iv_solved_from_mid`` so callers
    can avoid circular mid→IV→fair mispricing checks.
    """
    out = dict(leg)
    out.pop("_iv_solved_from_mid", None)
    strike = float(out.get("strike") or 0.0)
    if strike <= 0 or spot <= 0:
        return out

    t_years = _years_to_expiry(str(out.get("expiry") or ""))
    is_call = str(out.get("option_type") or "call").lower() == "call"
    mid = float(out.get("mid") or 0.0)
    iv = out.get("iv")
    iv_f = float(iv) if iv is not None and str(iv) not in ("", "nan") and pd.notna(iv) else math.nan
    solved_from_mid = False

    if not math.isfinite(iv_f) or iv_f <= 1e-6:
        if mid > 0 and math.isfinite(t_years) and t_years > 0:
            solved = solve_iv(
                market_price=mid,
                spot=spot,
                strike=strike,
                time_years=t_years,
                is_call=is_call,
                risk_free=risk_free,
            )
            if solved is not None and math.isfinite(solved) and solved > 1e-6:
                iv_f = float(solved)
                out["iv"] = iv_f
                solved_from_mid = True

    if not math.isfinite(iv_f) or iv_f <= 1e-6:
        return out

    if solved_from_mid:
        out["_iv_solved_from_mid"] = True

    g = full_greeks_row(
        spot=spot,
        strike=strike,
        time_years=t_years,
        iv=iv_f,
        risk_free=risk_free,
        is_call=is_call,
    )
    for key in ("delta", "gamma", "theta", "vega"):
        cur = out.get(key)
        if cur is None or (isinstance(cur, float) and not math.isfinite(cur)):
            val = getattr(g, key)
            if math.isfinite(val):
                out[key] = float(val)
    if "iv" not in out or out.get("iv") in (None, "", "nan"):
        out["iv"] = iv_f
    return out


def _leg_fair_mid(leg: dict[str, Any], *, spot: float, risk_free: float) -> float:
    strike = float(leg.get("strike") or 0.0)
    t_years = _years_to_expiry(str(leg.get("expiry") or ""))
    is_call = str(leg.get("option_type") or "call").lower() == "call"
    iv = leg.get("iv")
    iv_f = float(iv) if iv is not None and math.isfinite(float(iv)) else math.nan
    if not math.isfinite(iv_f) or iv_f <= 1e-6 or strike <= 0 or spot <= 0:
        return math.nan
    from rlm.data.microstructure.calculators.greeks import _bs_price

    try:
        return float(
            _bs_price(spot, strike, t_years, risk_free, iv_f, is_call)
        )
    except Exception:
        return math.nan


def assess_combo_edge(
    matched_legs: list[dict[str, Any]],
    *,
    spot: float,
    entry_debit_dollars: float,
    regime_direction: str = "",
    strategy_name: str = "",
    contract_multiplier: int = 100,
    risk_free: float = _DEFAULT_RISK_FREE,
) -> dict[str, Any]:
    """
    Compare BS fair combo mark vs market mid mark.

    ``buyer_edge_pct`` > 0 means market is cheaper than model (favorable for net
    debit buyers and for net-credit sellers when marks are signed consistently).

    Mispricing % is only computed from legs whose IV came from the chain (not
    solved from mid). Mid-implied IV makes fair≈mid by construction, which would
    false-reject every debit under a positive ``min_edge`` threshold. When no
    independent fair is available, the % gate is skipped (delta/spread still apply).
    """
    enriched = [enrich_leg_greeks(m, spot=spot, risk_free=risk_free) for m in matched_legs]
    market_mark = 0.0
    fair_mark = 0.0
    n_independent_fair = 0
    for leg in enriched:
        mid = float(leg.get("mid") or 0.0)
        fair = _leg_fair_mid(leg, spot=spot, risk_free=risk_free)
        sign = 1.0 if str(leg.get("side") or "").lower() == "long" else -1.0
        market_mark += sign * mid * contract_multiplier
        # Circular mid→IV→fair must not drive the % edge gate.
        if leg.pop("_iv_solved_from_mid", False):
            continue
        if math.isfinite(fair):
            fair_mark += sign * fair * contract_multiplier
            n_independent_fair += 1

    debit = float(entry_debit_dollars)
    is_credit = debit < 0
    buyer_edge_pct = math.nan
    # fair_mark stays 0.0 when nothing independent was added — that is "unknown",
    # not "worthless". Only score edge when at least one chain-IV fair exists.
    if n_independent_fair > 0 and abs(market_mark) > 1e-6:
        buyer_edge_pct = (fair_mark - market_mark) / abs(market_mark)

    is_short_dte = False
    if matched_legs:
        try:
            exp = str(matched_legs[0].get("expiry") or "")
            t_y = _years_to_expiry(exp)
            is_short_dte = math.isfinite(t_y) and t_y <= (6.0 / 365.0)
        except (TypeError, ValueError):
            pass
    min_edge = _env_float(
        "RLM_OPTIONS_MIN_BUYER_EDGE_PCT" if is_short_dte else "RLM_OPTIONS_SWING_MIN_BUYER_EDGE_PCT",
        0.02 if is_short_dte else 0.01,
    )

    passes = True
    reason = ""
    if math.isfinite(buyer_edge_pct):
        # Same signed convention for debit and credit: need buyer_edge >= min_edge.
        # (For credits with negative market_mark, positive edge means richer credit
        # than model — favorable to the seller.)
        if buyer_edge_pct < min_edge:
            kind = "credit_underpriced" if is_credit else "debit_overpriced"
            passes = False
            reason = f"{kind} edge={buyer_edge_pct:.2%} need>={min_edge:.2%}"

    dir_norm = str(regime_direction or "").strip().lower()
    net_delta = 0.0
    for leg in enriched:
        d = leg.get("delta")
        if d is None or not math.isfinite(float(d)):
            continue
        sign = 1.0 if str(leg.get("side") or "").lower() == "long" else -1.0
        net_delta += sign * float(d)

    strat = str(strategy_name or "").lower()
    neutral_structure = any(x in strat for x in ("straddle", "strangle", "iron", "condor", "butterfly"))
    delta_align = True
    if not neutral_structure and dir_norm == "bull" and net_delta < 0.12:
        delta_align = False
        reason = reason or "delta_misaligned_bull"
    elif not neutral_structure and dir_norm == "bear" and net_delta > -0.12:
        delta_align = False
        reason = reason or "delta_misaligned_bear"

    avg_gamma = _mean_finite([leg.get("gamma") for leg in enriched])
    avg_theta = _mean_finite([leg.get("theta") for leg in enriched])
    max_spread_pct = max(
        (
            (float(leg.get("ask") or 0) - float(leg.get("bid") or 0)) / float(leg.get("mid") or 1.0)
            for leg in enriched
            if float(leg.get("mid") or 0) > 0
        ),
        default=0.0,
    )
    max_spread_allowed = _env_float("RLM_OPTIONS_MAX_SPREAD_PCT_MID", 0.12)
    if max_spread_pct > max_spread_allowed:
        passes = False
        reason = reason or f"spread_too_wide {max_spread_pct:.1%}"

    return {
        "matched_legs_enriched": enriched,
        "spot": spot,
        "market_mark_dollars": market_mark,
        "fair_mark_dollars": fair_mark if n_independent_fair > 0 else math.nan,
        "buyer_edge_pct": buyer_edge_pct,
        "net_delta": net_delta,
        "avg_gamma": avg_gamma,
        "avg_theta": avg_theta,
        "max_spread_pct_mid": max_spread_pct,
        "passes_edge_gate": passes and delta_align,
        "edge_skip_reason": reason,
        "is_credit": is_credit,
    }


def _mean_finite(vals: list[Any]) -> float:
    xs = []
    for v in vals:
        try:
            f = float(v)
            if math.isfinite(f):
                xs.append(f)
        except (TypeError, ValueError):
            continue
    return sum(xs) / len(xs) if xs else math.nan


def chain_spot_price(chain: pd.DataFrame, fallback: float | None = None) -> float:
    for col in ("underlying_price", "spot", "underlying"):
        if col in chain.columns and not chain[col].empty:
            v = float(chain[col].dropna().iloc[-1])
            if math.isfinite(v) and v > 0:
                return v
    if fallback is not None and math.isfinite(fallback) and fallback > 0:
        return float(fallback)
    return math.nan
