from __future__ import annotations

import pandas as pd


def classify_direction(s_d: float) -> str:
    if pd.isna(s_d):
        return "unknown"
    if s_d > 0.6:
        return "bull"
    if s_d < -0.6:
        return "bear"
    if abs(s_d) < 0.3:
        return "range"
    return "transition"


def classify_volatility(s_v: float) -> str:
    if pd.isna(s_v):
        return "unknown"
    if s_v > 0.4:
        return "high_vol"
    if s_v < -0.4:
        return "low_vol"
    return "transition"


def classify_liquidity(s_l: float) -> str:
    if pd.isna(s_l):
        return "unknown"
    return "high_liquidity" if s_l > 0.4 else "low_liquidity"


def classify_dealer_flow(s_g: float) -> str:
    if pd.isna(s_g):
        return "unknown"
    return "supportive" if s_g > 0.4 else "destabilizing"
