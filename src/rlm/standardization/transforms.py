from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-12


def _scalar_finite(x: Any) -> bool:
    """True if ``x`` is a finite real scalar (handles ``pd.NA`` / nullable dtypes)."""
    try:
        if pd.isna(x):
            return False
    except TypeError:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def _clip_tanh_input(value: float, limit: float = 20.0) -> float:
    return float(np.clip(value, -limit, limit))


def log_tanh_ratio(
    x: float,
    x0: float,
    k: float = 1.0,
    invert: bool = False,
) -> float:
    """
    Standardize a positive ratio-like input to [-1, 1].

    Formula:
        s = tanh(k * ln(x / x0))

    If invert=True, the sign is flipped after transformation.
    Useful for factors where lower raw values are better, e.g. bid-ask spread.
    """
    if not _scalar_finite(x) or not _scalar_finite(x0):
        return np.nan
    xf, x0f = float(x), float(x0)
    if xf <= 0.0 or x0f <= 0.0:
        return np.nan

    z = k * math.log(max(xf, EPSILON) / max(x0f, EPSILON))
    s = math.tanh(_clip_tanh_input(z))
    return -s if invert else s


def log_tanh_signed(
    d: float,
    d0: float,
    k: float = 1.0,
    invert: bool = False,
) -> float:
    """
    Standardize a signed deviation-like input to [-1, 1].

    Formula:
        s = tanh(k * sign(d) * ln(1 + |d| / d0))
    """
    if not _scalar_finite(d) or not _scalar_finite(d0):
        return np.nan
    df, d0f = float(d), float(d0)
    if d0f <= 0.0:
        return np.nan

    z = k * np.sign(df) * math.log(1.0 + abs(df) / max(d0f, EPSILON))
    s = math.tanh(_clip_tanh_input(z))
    return -s if invert else s


def sigma_floor(value: float, minimum: float) -> float:
    if not _scalar_finite(value):
        return minimum
    return max(float(value), float(minimum))
