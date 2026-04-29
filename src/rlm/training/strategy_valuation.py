from __future__ import annotations

import numpy as np

from rlm.training.strategy_structures import SpreadStructure


def value_bull_call_at_expiry(structure: SpreadStructure, final_price: float) -> float:
    assert structure.long_strike is not None
    assert structure.short_strike is not None
    long_intrinsic = max(final_price - structure.long_strike, 0.0)
    short_intrinsic = max(final_price - structure.short_strike, 0.0)
    gross_value = long_intrinsic - short_intrinsic
    return float(min(gross_value, structure.width_abs))


def value_bear_put_at_expiry(structure: SpreadStructure, final_price: float) -> float:
    assert structure.long_strike is not None
    assert structure.short_strike is not None
    long_intrinsic = max(structure.long_strike - final_price, 0.0)
    short_intrinsic = max(structure.short_strike - final_price, 0.0)
    gross_value = long_intrinsic - short_intrinsic
    return float(min(gross_value, structure.width_abs))


def value_iron_condor_path(
    structure: SpreadStructure,
    path: np.ndarray,
    *,
    use_path_exits: bool = True,
) -> float:
    assert structure.lower_short is not None
    assert structure.upper_short is not None

    breaches = ((path < structure.lower_short) | (path > structure.upper_short)).astype(float)
    breach_ratio = float(breaches.mean()) if len(breaches) else 0.0
    gross_credit_value = max(0.0, structure.entry_price * (1.0 - 1.5 * breach_ratio))
    if use_path_exits and breach_ratio > 0.50:
        gross_credit_value *= 0.5
    return float(gross_credit_value)


def value_calendar_path(
    structure: SpreadStructure,
    path: np.ndarray,
    *,
    entry_sigma: float,
    realized_vol: float,
    horizon_fraction: float,
) -> float:
    center = structure.center_price
    displacement = abs(float(path[-1]) - center) / max(center, 1e-6)
    vol_edge = (realized_vol - entry_sigma) / max(entry_sigma, 1e-6)
    theta_decay = 0.35 * horizon_fraction
    displacement_penalty = 0.9 * displacement
    gross_value = structure.entry_price * (1.0 + 1.25 * vol_edge - theta_decay - displacement_penalty)
    return float(max(gross_value, 0.0))


def value_debit_spread_path(
    structure: SpreadStructure,
    path: np.ndarray,
    *,
    trend_strength: float,
) -> float:
    assert structure.long_strike is not None
    assert structure.short_strike is not None

    final_price = float(path[-1])
    direction = 1.0 if structure.short_strike >= structure.long_strike else -1.0
    realized_direction = np.sign(final_price - structure.center_price)
    trend_direction = np.sign(trend_strength) if trend_strength != 0 else direction

    directional_move = abs(final_price - structure.center_price)
    gross_value = min(directional_move, structure.width_abs)
    if realized_direction == 0:
        gross_value *= 0.5
    elif realized_direction != trend_direction:
        gross_value *= 0.25
    return float(max(gross_value, 0.0))
