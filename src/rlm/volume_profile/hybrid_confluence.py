"""Hybrid confluence scoring across VP, GEX, and IV surface context."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from rlm.data.microstructure.database.query import MicrostructureDB


def _to_utc(ts: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def vp_gex_confluence(
    symbol: str, timestamp: datetime, vp_levels: list[float]
) -> dict[float, dict[str, float]]:
    """Map VP levels to nearest-strike GEX metrics and confluence scores."""
    ts = _to_utc(timestamp)
    try:
        db = MicrostructureDB()
    except Exception:
        return {
            float(level): {
                "net_gex": float("nan"),
                "gex_percentile": float("nan"),
                "confluence_score": 0.0,
            }
            for level in vp_levels
        }

    gex = db.load_gex_surface(
        symbol.upper(), ts.date().isoformat(), ts.date().isoformat(), net_gex_only=False
    )
    if gex.empty:
        return {
            float(level): {
                "net_gex": float("nan"),
                "gex_percentile": float("nan"),
                "confluence_score": 0.0,
            }
            for level in vp_levels
        }

    gex = gex.copy()
    gex["timestamp"] = pd.to_datetime(gex["timestamp"], utc=True, errors="coerce")
    gex = gex.loc[gex["timestamp"] <= ts]
    if gex.empty:
        return {
            float(level): {
                "net_gex": float("nan"),
                "gex_percentile": float("nan"),
                "confluence_score": 0.0,
            }
            for level in vp_levels
        }

    snap = gex.loc[gex["timestamp"] == gex["timestamp"].max()].copy()
    abs_gex = snap["net_gex"].abs()

    out: dict[float, dict[str, float]] = {}
    for level in vp_levels:
        nearest = snap.iloc[(snap["strike"] - float(level)).abs().argsort().iloc[0]]
        net_gex = float(nearest.get("net_gex", float("nan")))
        percentile = (
            float((abs_gex.rank(pct=True)).loc[nearest.name]) if len(abs_gex) else float("nan")
        )
        score = float(np.tanh(abs(net_gex) / (abs_gex.max() + 1e-9)) * percentile)
        out[float(level)] = {
            "net_gex": net_gex,
            "gex_percentile": percentile,
            "confluence_score": score,
        }
    return out


def iv_surface_at_vp_levels(
    symbol: str, timestamp: datetime, vp_levels: list[float]
) -> dict[float, dict[str, float]]:
    """Return IV metrics sampled at moneyness implied by each VP level."""
    ts = _to_utc(timestamp)
    try:
        db = MicrostructureDB()
    except Exception:
        return {
            float(level): {
                "iv": float("nan"),
                "iv_skew": float("nan"),
                "iv_term_structure": float("nan"),
            }
            for level in vp_levels
        }

    iv = db.load_iv_surface_range(symbol.upper(), ts.date().isoformat(), ts.date().isoformat())
    if iv.empty:
        return {
            float(level): {
                "iv": float("nan"),
                "iv_skew": float("nan"),
                "iv_term_structure": float("nan"),
            }
            for level in vp_levels
        }

    iv = iv.copy()
    iv["timestamp"] = pd.to_datetime(iv["timestamp"], utc=True, errors="coerce")
    iv = iv.loc[iv["timestamp"] <= ts]
    if iv.empty:
        return {
            float(level): {
                "iv": float("nan"),
                "iv_skew": float("nan"),
                "iv_term_structure": float("nan"),
            }
            for level in vp_levels
        }

    snap = iv.loc[iv["timestamp"] == iv["timestamp"].max()].copy()
    spot = (
        float(snap.get("underlying_price", pd.Series([1.0])).dropna().iloc[0])
        if "underlying_price" in snap
        else 1.0
    )
    spot = spot if np.isfinite(spot) and spot > 0 else 1.0

    out: dict[float, dict[str, float]] = {}
    for level in vp_levels:
        moneyness = float(level) / spot
        near = snap.iloc[(snap["moneyness"] - moneyness).abs().argsort().iloc[0]]
        iv_val = float(near.get("implied_vol", float("nan")))
        same_dte = snap.loc[snap["days_to_expiry"] == near["days_to_expiry"]]
        skew = (
            float(same_dte["implied_vol"].max() - same_dte["implied_vol"].min())
            if not same_dte.empty
            else float("nan")
        )
        near_money = snap.loc[(snap["moneyness"] - moneyness).abs() < 0.03].sort_values(
            "days_to_expiry"
        )
        term = (
            float(near_money["implied_vol"].iloc[0] / near_money["implied_vol"].iloc[-1])
            if len(near_money) >= 2 and near_money["implied_vol"].iloc[-1] > 0
            else float("nan")
        )
        out[float(level)] = {"iv": iv_val, "iv_skew": skew, "iv_term_structure": term}
    return out


def hybrid_support_resistance(
    symbol: str, timestamp: datetime, vp_profile: dict[str, Any]
) -> pd.DataFrame:
    """Combine VP levels with GEX and IV context into level-level strength scores."""
    levels = [
        vp_profile.get("poc"),
        vp_profile.get("value_area_high"),
        vp_profile.get("value_area_low"),
    ]
    levels.extend(vp_profile.get("hvn_levels", []) or [])
    levels.extend(vp_profile.get("lvn_levels", []) or [])
    levels = [float(x) for x in levels if x is not None and np.isfinite(float(x))]
    if not levels:
        return pd.DataFrame(
            columns=[
                "level",
                "net_gex",
                "gex_percentile",
                "iv",
                "iv_skew",
                "iv_term_structure",
                "strength_score",
            ]
        )

    gex_map = vp_gex_confluence(symbol, timestamp, levels)
    iv_map = iv_surface_at_vp_levels(symbol, timestamp, levels)

    rows = []
    for level in levels:
        gex_data = gex_map.get(level, {})
        iv_data = iv_map.get(level, {})
        gex_score = float(gex_data.get("confluence_score", 0.0))
        iv_score = (
            0.0
            if pd.isna(iv_data.get("iv_skew", np.nan))
            else float(np.tanh(abs(float(iv_data["iv_skew"]))))
        )
        strength = float(np.clip(0.65 * gex_score + 0.35 * iv_score, 0.0, 1.0))
        rows.append(
            {
                "level": level,
                "net_gex": gex_data.get("net_gex", float("nan")),
                "gex_percentile": gex_data.get("gex_percentile", float("nan")),
                "iv": iv_data.get("iv", float("nan")),
                "iv_skew": iv_data.get("iv_skew", float("nan")),
                "iv_term_structure": iv_data.get("iv_term_structure", float("nan")),
                "strength_score": strength,
            }
        )
    return pd.DataFrame(rows).sort_values("strength_score", ascending=False).reset_index(drop=True)
