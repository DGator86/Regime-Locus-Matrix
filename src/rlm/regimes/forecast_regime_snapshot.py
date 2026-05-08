"""Serialize live forecast regime / transition-matrix fields for universe JSON and equity exits."""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def state_label_bull_alignment_weight(label: str) -> float:
    """Map HMM/Markov state label → [0,1] bullish equity thesis alignment (deterministic heuristic)."""
    s = str(label).lower().replace(" ", "_")
    if ("trend_down" in s) or ("down_stable" in s) or ("_bear" in s or s.endswith("bear")):
        return 0.0
    if "bear" in s and "bull" not in s:
        return 0.0
    if ("trend_up" in s) or ("up_stable" in s):
        return 1.0
    if "bull" in s:
        return 1.0
    if any(k in s for k in ("range", "mean_revert", "compression", "condor")):
        return 0.5
    if "breakout" in s:
        return 0.65
    if any(k in s for k in ("transition", "chaos", "no_trade")):
        return 0.5
    return 0.5


def label_aligned_transition_masses(
    next_probs: Sequence[float],
    state_labels: Sequence[str],
) -> tuple[float | None, float | None]:
    """Σ P(next=j) × w_j  (bull) and Σ × (1 − w_j) (bear); requires aligned lengths."""
    if not next_probs or not state_labels or len(next_probs) != len(state_labels):
        return None, None
    ps = []
    ws = []
    for p, lbl in zip(next_probs, state_labels):
        try:
            fp = float(p)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(fp) or fp < 0.0:
            return None, None
        ps.append(fp)
        ws.append(float(state_label_bull_alignment_weight(str(lbl))))
    total = sum(ps)
    if total <= 0.0:
        return None, None
    ps = [p / total for p in ps]
    bull_mass = float(sum(p * w for p, w in zip(ps, ws)))
    bear_mass = float(sum(p * (1.0 - w) for p, w in zip(ps, ws)))
    bull_mass = min(1.0, max(0.0, bull_mass))
    bear_mass = min(1.0, max(0.0, bear_mass))
    return bull_mass, bear_mass


def position_directional_transition_mass(snapshot: Mapping[str, Any] | None, position_direction: str) -> float | None:
    """Precomputed masses in snapshot → mass supporting the traded direction."""
    if not snapshot:
        return None
    d = str(position_direction or "").strip().lower()
    if d not in ("bull", "bear"):
        return None
    bm = snapshot.get("next_label_aligned_bull_mass")
    brm = snapshot.get("next_label_aligned_bear_mass")
    try:
        if d == "bull":
            return float(bm) if bm is not None else None
        return float(brm) if brm is not None else None
    except (TypeError, ValueError):
        return None


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


def _empty_transition_payload() -> dict[str, Any]:
    return {
        "family": "none",
        "most_likely_next_prob": None,
        "most_likely_next_prob_calibrated": None,
        "most_likely_next_state": None,
        "most_likely_next_label": None,
        "expected_persistence": None,
        "transition_entropy": None,
        "next_probs": None,
        "state_labels": None,
        "next_label_aligned_bull_mass": None,
        "next_label_aligned_bear_mass": None,
        "current_state_label": None,
    }


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

    def _labels_vocab(prefix: str) -> list[str] | None:
        col = f"{prefix}_state_labels"
        if col not in last.index:
            return None
        raw = last.get(col)
        if raw is None:
            return None
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)) and raw:
            if all(isinstance(x, str) for x in raw):
                return [str(x) for x in raw]
        return None

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
        return _empty_transition_payload()

    cal_key = f"{prefix}_most_likely_next_prob_calibrated"
    cal_v = _flt(cal_key) if cal_key in last.index else None

    state_lbl_key = f"{prefix}_state_label"
    cur_lbl = _lbl(state_lbl_key)
    probs = _probs(prefix)
    labels = _labels_vocab(prefix)
    bull_m = bear_m = None
    if probs is not None and labels is not None:
        bull_m, bear_m = label_aligned_transition_masses(probs, labels)

    return {
        "family": fam,
        "most_likely_next_prob": _flt(f"{prefix}_most_likely_next_prob"),
        "most_likely_next_prob_calibrated": cal_v,
        "most_likely_next_state": _nint(f"{prefix}_most_likely_next_state"),
        "most_likely_next_label": _lbl(f"{prefix}_most_likely_next_label"),
        "expected_persistence": _flt(f"{prefix}_expected_persistence"),
        "transition_entropy": _flt(f"{prefix}_regime_transition_entropy"),
        "next_probs": probs,
        "state_labels": labels,
        "next_label_aligned_bull_mass": bull_m,
        "next_label_aligned_bear_mass": bear_m,
        "current_state_label": cur_lbl,
    }
