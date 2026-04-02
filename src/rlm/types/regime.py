from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeState:
    direction: str
    volatility: str
    liquidity: str
    dealer_flow: str
    regime_key: str
