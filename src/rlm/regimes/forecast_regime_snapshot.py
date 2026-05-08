"""Serialize live forecast regime / transition-matrix fields for universe JSON and equity exits."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def regime_direction_equity(regime_key: str) -> str:
    """First direction token of ``direction|vol|liq|dealer`` ROEE keys (lowercase)."""
    rk = str(regime_key or "").strip()
    if "|" in rk:
        return str(rk.split("|")[0]).strip().lower()
    return rk.lower()


def plan_regime_key(plan_row: Mapping[str, Any]) -> str:
    """Resolve regime_key from an active-universe row (top-level → decision → pipeline)."""
    top = plan_row.get("regime_key")
    if isinstance(top, str) and top.strip():
        return top.strip()
    dec = plan_row.get("decision") if isinstance(plan_row.get("decision"), dict) else {}
    rk = dec.get("regime_key") if isinstance(dec, dict) else ""
    if isinstance(rk, str) and rk.strip():
        return rk.strip()
    pipe = plan_row.get("pipeline") if isinstance(plan_row.get("pipeline"), dict) else {}
    rk2 = pipe.get("regime_key") if isinstance(pipe, dict) else ""
    if isinstance(rk2, str):
        return rk2.strip()
    return ""


def regime_transition_best_prob(snapshot: Mapping[str, Any] | None) -> float | None:
    """Prefer calibrated top-1 next-step probability when present."""
    if not snapshot:
        return None
    cal = snapshot.get("most_likely_next_prob_calibrated")
    raw = snapshot.get("most_likely_next_prob")
    chosen: Any = None
    if cal is not None and pd.notna(cal):
        chosen = cal
    elif raw is not None and pd.notna(raw):
        chosen = raw
    if chosen is None:
        return None
    try:
        val = float(chosen)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return float(val)


def build_regime_transition_snapshot(last: pd.Series, *, live_model: str = "forecast") -> dict[str, Any]:
    """Subset of HMM / Markov one-step predictive fields suitable for JSON (one bar)."""
    lm = str(live_model or "").lower().strip()

    def _flt(k: str) -> float | None:
        if k not in last.index:
            return None
        v = last.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def _nint(k: str) -> int | None:
        if k not in last.index:
            return None
        v = last.get(k)
        if v is None or pd.isna(v):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _lbl(k: str) -> str | None:
        if k not in last.index:
            return None
        v = last.get(k)
        if v is None or pd.isna(v):
            return None
        s = str(v).strip()
        return s if s else None

    def _probs(prefix: str) -> list[float] | None:
        col = f"{prefix}_next_probs"
        if col not in last.index:
            return None
        raw = last.get(col)
        if raw is None:
            return None
        if isinstance(raw, np.ndarray):
            seq = raw.astype(float).tolist()
        elif isinstance(raw, (list, tuple)):
            seq = [float(x) for x in raw]
        else:
            return None
        if not seq:
            return None
        out: list[float] = []
        for fx in seq:
            if math.isfinite(fx):
                out.append(fx)
        return out if out else None

    hmm_ok = _flt("hmm_most_likely_next_prob") is not None
    markov_ok = _flt("markov_most_likely_next_prob") is not None
    prefix: str
    fam: str
    if lm == "markov" and markov_ok:
        prefix, fam = "markov", "markov"
    elif lm == "hmm" and hmm_ok:
        prefix, fam = "hmm", "hmm"
    elif hmm_ok:
        prefix, fam = "hmm", "hmm"
    elif markov_ok:
        prefix, fam = "markov", "markov"
    else:
        return {
            "family": "none",
            "most_likely_next_prob": None,
            "most_likely_next_prob_calibrated": None,
            "most_likely_next_state": None,
            "most_likely_next_label": None,
            "expected_persistence": None,
            "transition_entropy": None,
            "next_probs": None,
            "current_state_label": None,
        }

    cal_key = f"{prefix}_most_likely_next_prob_calibrated"
    cal_v = _flt(cal_key) if cal_key in last.index else None

    state_lbl_key = f"{prefix}_state_label"
    cur_lbl = _lbl(state_lbl_key)

    return {
        "family": fam,
        "most_likely_next_prob": _flt(f"{prefix}_most_likely_next_prob"),
        "most_likely_next_prob_calibrated": cal_v,
        "most_likely_next_state": _nint(f"{prefix}_most_likely_next_state"),
        "most_likely_next_label": _lbl(f"{prefix}_most_likely_next_label"),
        "expected_persistence": _flt(f"{prefix}_expected_persistence"),
        "transition_entropy": _flt(f"{prefix}_regime_transition_entropy"),
        "next_probs": _probs(prefix),
        "current_state_label": cur_lbl,
    }
