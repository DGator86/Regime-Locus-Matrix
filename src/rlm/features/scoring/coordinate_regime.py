from __future__ import annotations

import math
from typing import Literal, Mapping

RegimeLabel = Literal[
    "trend_up_stable",
    "trend_down_stable",
    "range_compression",
    "mean_reversion",
    "breakout_expansion",
    "transition",
    "chaos",
    "no_trade",
]


def classify_regime_from_coordinates(row: Mapping[str, float]) -> RegimeLabel:
    try:
        D = float(row["M_D"])
        V = float(row["M_V"])
        L = float(row["M_L"])
        G = float(row["M_G"])
        delta = float(row["M_delta_neutral"])
    except (TypeError, ValueError):
        return "no_trade"
    R_trans_raw = row.get("M_R_trans", 0.0)
    try:
        R_trans = float(R_trans_raw) if R_trans_raw is not None else 0.0
    except (TypeError, ValueError):
        R_trans = 0.0
    if not all(math.isfinite(v) for v in (D, V, L, G, delta)):
        return "no_trade"
    if not math.isfinite(R_trans):
        R_trans = 0.0

    d = D - 5.0
    g = G - 5.0
    trend_strength = abs(d)
    alignment = d * g
    _ = delta  # reserved for future thresholding

    # 1. Hard execution filter
    if L < 3.0:
        return "no_trade"
    # 2. Chaos regime
    if V > 7.0 and L < 5.0:
        return "chaos"
    # 3. Transition regime (highest priority)
    if R_trans > 4.0:
        return "transition"
    # 4. Range / compression
    if trend_strength < 1.5 and V < 4.0 and g > 0.0:
        return "range_compression"
    # 5. Mean reversion (dealer opposition)
    if alignment < -5.0:
        return "mean_reversion"
    # 6. Stable trends
    if alignment > 5.0:
        if d > 0.0:
            return "trend_up_stable"
        return "trend_down_stable"
    # 7. Breakout / expansion
    if V > 6.0:
        return "breakout_expansion"
    return "no_trade"
