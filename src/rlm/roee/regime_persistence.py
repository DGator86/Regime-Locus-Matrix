from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimePersistenceState:
    regime: str
    consecutive_bars: int
    dominant_probability: float
    flip_rate_recent: float


def persistence_size_multiplier(state: RegimePersistenceState) -> float:
    if state.flip_rate_recent > 0.50:
        return 0.50
    if state.consecutive_bars >= 5 and state.dominant_probability >= 0.65:
        return 1.15
    if state.consecutive_bars <= 1:
        return 0.75
    return 1.0


def persistence_trade_gate(state: RegimePersistenceState) -> bool:
    if state.flip_rate_recent > 0.70:
        return False
    if state.dominant_probability < 0.40:
        return False
    return True
