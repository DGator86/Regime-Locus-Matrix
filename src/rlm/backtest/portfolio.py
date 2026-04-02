from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rlm.roee.chain_match import estimate_entry_cost_from_matched_legs, estimate_mark_value_from_matched_legs
from rlm.types.options import TradeDecision
from rlm.types.results import TradeRecord


@dataclass
class OpenPosition:
    position_id: str
    timestamp_entry: pd.Timestamp
    strategy_name: str
    regime_key: str
    underlying_symbol: str
    entry_underlying_price: float
    expiry: str | None
    entry_cost: float
    quantity: int
    matched_legs: list[dict]
    target_profit_pct: float
    max_risk_pct: float
    metadata: dict = field(default_factory=dict)

    def current_mark_value(self) -> float:
        return estimate_mark_value_from_matched_legs(self.matched_legs)

    def pnl(self) -> float:
        # debit: entry_cost positive. current value positive if favorable
        return self.current_mark_value() - self.entry_cost

    def pnl_pct(self) -> float:
        denom = abs(self.entry_cost) if abs(self.entry_cost) > 1e-9 else np.nan
        return self.pnl() / denom if pd.notna(denom) else np.nan


class Portfolio:
    def __init__(self, initial_capital: float = 100_000.0, contract_multiplier: int = 100) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.contract_multiplier = contract_multiplier
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[TradeRecord] = []
        self.equity_history: list[dict] = []

    def available_cash(self) -> float:
        return self.cash

    def total_mark_value(self) -> float:
        return sum(pos.current_mark_value() * pos.quantity for pos in self.open_positions.values())

    def equity(self) -> float:
        return self.cash + self.total_mark_value()

    def can_open(self, entry_cost: float, quantity: int = 1) -> bool:
        required = max(entry_cost, 0.0) * quantity
        return self.cash >= required

    def open_from_decision(
        self,
        *,
        timestamp: pd.Timestamp,
        underlying_symbol: str,
        underlying_price: float,
        decision: TradeDecision,
        quantity: int = 1,
    ) -> str | None:
        if decision.action != "enter":
            return None

        entry_cost = estimate_entry_cost_from_matched_legs(
            decision=decision,
            contract_multiplier=self.contract_multiplier,
        )
        if not np.isfinite(entry_cost):
            return None

        total_entry_cost = entry_cost * quantity
        if not self.can_open(total_entry_cost, quantity=1):
            return None

        self.cash -= max(total_entry_cost, 0.0)

        matched = decision.metadata.get("matched_legs", [])
        expiry = matched[0]["expiry"] if matched else None
        position_id = str(uuid.uuid4())

        self.open_positions[position_id] = OpenPosition(
            position_id=position_id,
            timestamp_entry=pd.Timestamp(timestamp),
            strategy_name=decision.strategy_name or "unknown",
            regime_key=decision.regime_key or "unknown",
            underlying_symbol=underlying_symbol,
            entry_underlying_price=float(underlying_price),
            expiry=expiry,
            entry_cost=float(entry_cost),
            quantity=int(quantity),
            matched_legs=matched,
            target_profit_pct=float(decision.target_profit_pct or 0.5),
            max_risk_pct=float(decision.max_risk_pct or 0.02),
            metadata=dict(decision.metadata),
        )
        return position_id

    def close_position(
        self,
        *,
        position_id: str,
        timestamp_exit: pd.Timestamp,
        underlying_price: float,
        exit_reason: str,
    ) -> TradeRecord | None:
        pos = self.open_positions.get(position_id)
        if pos is None:
            return None

        exit_value = pos.current_mark_value()
        total_exit_value = exit_value * pos.quantity
        self.cash += max(total_exit_value, 0.0)

        pnl = (exit_value - pos.entry_cost) * pos.quantity
        pnl_pct = pnl / (abs(pos.entry_cost) * pos.quantity) if abs(pos.entry_cost) > 1e-9 else np.nan

        trade = TradeRecord(
            position_id=pos.position_id,
            timestamp_entry=str(pos.timestamp_entry),
            timestamp_exit=str(pd.Timestamp(timestamp_exit)),
            strategy_name=pos.strategy_name,
            regime_key=pos.regime_key,
            underlying_symbol=pos.underlying_symbol,
            entry_underlying_price=pos.entry_underlying_price,
            exit_underlying_price=float(underlying_price),
            entry_cost=pos.entry_cost * pos.quantity,
            exit_value=exit_value * pos.quantity,
            pnl=float(pnl),
            pnl_pct=float(pnl_pct) if pd.notna(pnl_pct) else np.nan,
            quantity=pos.quantity,
            exit_reason=exit_reason,
            metadata=dict(pos.metadata),
        )

        self.closed_trades.append(trade)
        del self.open_positions[position_id]
        return trade

    def mark_equity(self, timestamp: pd.Timestamp) -> None:
        self.equity_history.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "cash": self.cash,
                "mark_value": self.total_mark_value(),
                "equity": self.equity(),
                "open_positions": len(self.open_positions),
            }
        )

    def equity_frame(self) -> pd.DataFrame:
        if not self.equity_history:
            return pd.DataFrame(columns=["timestamp", "cash", "mark_value", "equity", "open_positions"])
        return pd.DataFrame(self.equity_history).set_index("timestamp").sort_index()

    def closed_trades_frame(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.closed_trades])
