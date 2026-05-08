"""Regime-derived directional hint for ``rlm challenge`` when persona vets to ``no_trade``.

The persona pipeline is intentionally conservative (trap / alignment gates). Challenge
sessions still benefit from showcasing RLM directional bias derived from latent regime
labels on the ROEE policy row — without relaxing real-money gates elsewhere.
"""

from __future__ import annotations

import os

import pandas as pd

from rlm.roee.decision import resolve_latent_regime_from_row
from rlm.roee.sizing import parse_latent_regime_label


def regime_fallback_disabled() -> bool:
    v = os.environ.get("RLM_CHALLENGE_REGIME_FALLBACK", "1").strip().lower()
    return v in {"0", "false", "no", "off", "disabled"}


def regime_fallback_floor_scores() -> tuple[float, float]:
    """Minimum (signal_alignment, confidence) when fallback fires."""
    try:
        a = float((os.environ.get("RLM_CHALLENGE_FALLBACK_ALIGNMENT") or "0.72").strip())
    except ValueError:
        a = 0.72
    try:
        c = float((os.environ.get("RLM_CHALLENGE_FALLBACK_CONFIDENCE") or "0.72").strip())
    except ValueError:
        c = 0.72
    return max(0.0, min(1.0, a)), max(0.0, min(1.0, c))


def _directive_from_latent_row(policy_row: pd.Series) -> tuple[str | None, str]:
    latent = resolve_latent_regime_from_row(policy_row)
    label_raw = latent.get("label")
    if label_raw is None or (isinstance(label_raw, float) and pd.isna(label_raw)):  # type: ignore[arg-type]
        return None, "no latent regime label"
    label = str(label_raw).strip()
    low = label.lower()

    direc, _vol = parse_latent_regime_label(low)
    if direc == "bull":
        return "long", f"fallback pipe bull ({label[:48]})"
    if direc == "bear":
        return "short", f"fallback pipe bear ({label[:48]})"

    chop_tokens = ("range", "mean_revert", "compression", "transition", "neutral")
    if any(t in low for t in chop_tokens):
        return None, f"fallback suppressed (choppy label: {label[:48]})"

    if "trend_down" in low or low.startswith("bear"):
        return "short", f"fallback label bearish ({label[:48]})"
    if "trend_up" in low or low.startswith("bull"):
        return "long", f"fallback label bullish ({label[:48]})"

    return None, f"no directional fallback for label ({label[:48]})"


def regime_fallback_directive_from_policy_row(
    policy_row: pd.Series,
) -> tuple[str | None, str]:
    """Map last ROEE policy row → ``long`` / ``short`` using latent + coarse regimes."""
    if regime_fallback_disabled():
        return None, "regime fallback disabled"

    d, note = _directive_from_latent_row(policy_row)
    if d is not None:
        return d, note

    if "direction_regime" in policy_row.index and pd.notna(policy_row.get("direction_regime")):
        dr = str(policy_row["direction_regime"]).strip().lower()
        if dr == "bull":
            return "long", "fallback direction_regime=bull"
        if dr == "bear":
            return "short", "fallback direction_regime=bear"

    rk = policy_row.get("regime_key")
    if rk is not None and not (isinstance(rk, float) and pd.isna(rk)):  # type: ignore[arg-type]
        head = str(rk).strip().lower().split("|", 1)[0].strip()
        chop = {"range", "transition", "neutral"}
        if head in chop:
            return None, f"fallback suppressed regime_key={head}"
        if head == "bull":
            return "long", "fallback regime_key pipe bull"
        if head == "bear":
            return "short", "fallback regime_key pipe bear"

    return None, "no regime fallback"
