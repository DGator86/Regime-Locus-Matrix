from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PositionRecord:
    position_id: str
    timestamp: str
    strategy_name: str
    regime_key: str
    underlying_symbol: str
    entry_underlying_price: float
    expiry: str | None
    entry_cost: float
    max_risk: float
    quantity: int
    status: str = "open"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    position_id: str
    timestamp_entry: str
    timestamp_exit: str
    strategy_name: str
    regime_key: str
    underlying_symbol: str
    entry_underlying_price: float
    exit_underlying_price: float
    entry_cost: float
    exit_value: float
    pnl: float
    pnl_pct: float
    quantity: int
    exit_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
