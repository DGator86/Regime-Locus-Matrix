from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OptionLeg:
    side: str  # "long" | "short"
    option_type: str  # "call" | "put"
    strike: float
    expiry: str | None = None
    quantity: int = 1


@dataclass(frozen=True)
class TradeCandidate:
    strategy_name: str
    regime_key: str
    rationale: str
    target_dte_min: int
    target_dte_max: int
    target_profit_pct: float
    max_risk_pct: float
    wings_sigma_low: float | None = None
    wings_sigma_high: float | None = None
    long_sigma: float | None = None
    short_sigma: float | None = None
    hedge_sigma: float | None = None
    defined_risk: bool = True


@dataclass
class TradeDecision:
    action: str  # "enter" | "hold" | "skip" | "exit"
    strategy_name: str | None = None
    regime_key: str | None = None
    rationale: str | None = None
    size_fraction: float | None = None
    target_profit_pct: float | None = None
    max_risk_pct: float | None = None
    candidate: TradeCandidate | None = None
    legs: list[OptionLeg] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
