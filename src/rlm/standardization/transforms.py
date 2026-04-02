from __future__ import annotations

import math

import numpy as np

EPSILON = 1e-12


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
    if not np.isfinite(x) or not np.isfinite(x0):
        return np.nan
    if x <= 0.0 or x0 <= 0.0:
        return np.nan

    z = k * math.log(max(x, EPSILON) / max(x0, EPSILON))
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
    if not np.isfinite(d) or not np.isfinite(d0):
        return np.nan
    if d0 <= 0.0:
        return np.nan

    z = k * np.sign(d) * math.log(1.0 + abs(d) / max(d0, EPSILON))
    s = math.tanh(_clip_tanh_input(z))
    return -s if invert else s


def sigma_floor(value: float, minimum: float) -> float:
    if not np.isfinite(value):
        return minimum
    return max(float(value), float(minimum))
