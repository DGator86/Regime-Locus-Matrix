from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rlm.backtest.fills import FillConfig, entry_fill_price, signed_cashflow_for_fill
from rlm.backtest.lifecycle import LifecycleConfig
from rlm.backtest.revalue import (
    aggregate_repriced_exit_value,
    aggregate_repriced_mark_value,
    has_full_reprice,
    reprice_matched_legs,
)
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
    entry_bar_index: int | None = None
    current_mark_value_cache: float | None = None
    current_exit_value_cache: float | None = None

    def mark_value(self) -> float:
        if self.current_mark_value_cache is None:
            return 0.0
        return float(self.current_mark_value_cache)

    def exit_value(self) -> float:
        if self.current_exit_value_cache is None:
            return 0.0
        return float(self.current_exit_value_cache)

    def pnl(self) -> float:
        return self.exit_value() - self.entry_cost

    def pnl_pct(self) -> float:
        denom = abs(self.entry_cost) if abs(self.entry_cost) > 1e-9 else np.nan
        return self.pnl() / denom if pd.notna(denom) else np.nan


class Portfolio:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        contract_multiplier: int = 100,
        lifecycle_config: LifecycleConfig | None = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.contract_multiplier = contract_multiplier
        self.lifecycle_config = lifecycle_config or LifecycleConfig()
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[TradeRecord] = []
        self.equity_history: list[dict] = []

    def available_cash(self) -> float:
        return self.cash

    def total_mark_value(self) -> float:
        return sum(pos.mark_value() * pos.quantity for pos in self.open_positions.values())

    def equity(self) -> float:
        return self.cash + self.total_mark_value()

    def can_open(self, entry_cost: float, quantity: int = 1) -> bool:
        required = max(entry_cost, 0.0) * quantity
        return self.cash >= required

    def _compute_entry_cost(
        self,
        *,
        matched_legs: list[dict],
        fill_config: FillConfig | None = None,
    ) -> float:
        cfg = fill_config or FillConfig(contract_multiplier=self.contract_multiplier)

        total = 0.0
        for leg in matched_legs:
            premium = entry_fill_price(
                side=str(leg["side"]),
                bid=float(leg["bid"]),
                ask=float(leg["ask"]),
                config=cfg,
            )
            total += signed_cashflow_for_fill(
                side=str(leg["side"]),
                premium=premium,
                contract_multiplier=self.contract_multiplier,
            )
        return float(total)

    def open_from_decision(
        self,
        *,
        timestamp: pd.Timestamp,
        underlying_symbol: str,
        underlying_price: float,
        decision: TradeDecision,
        quantity: int = 1,
        fill_config: FillConfig | None = None,
        bar_index: int | None = None,
    ) -> str | None:
        if decision.action != "enter":
            return None

        matched = decision.metadata.get("matched_legs", [])
        if not matched:
            return None

        entry_cost = self._compute_entry_cost(matched_legs=matched, fill_config=fill_config)
        leg_count = max(len(matched), 1)
        entry_commission = self.lifecycle_config.commission_per_contract * leg_count * quantity
        total_entry_cost = entry_cost * quantity

        if not self.can_open(total_entry_cost + entry_commission, quantity=1):
            return None

        self.cash -= max(total_entry_cost, 0.0)
        self.cash -= entry_commission

        expiry = matched[0]["expiry"] if matched else None
        position_id = str(uuid.uuid4())

        pos = OpenPosition(
            position_id=position_id,
            timestamp_entry=pd.Timestamp(timestamp),
            strategy_name=decision.strategy_name or "unknown",
            regime_key=decision.regime_key or "unknown",
            underlying_symbol=underlying_symbol,
            entry_underlying_price=float(underlying_price),
            expiry=expiry,
            entry_cost=float(entry_cost + (entry_commission / quantity)),
            quantity=int(quantity),
            matched_legs=matched,
            target_profit_pct=float(decision.target_profit_pct or 0.5),
            max_risk_pct=float(decision.max_risk_pct or 0.02),
            metadata=dict(decision.metadata),
            entry_bar_index=bar_index,
        )

        # initialize mark and exit caches using entry snapshot
        pos.current_mark_value_cache = 0.0
        pos.current_exit_value_cache = 0.0

        self.open_positions[position_id] = pos
        return position_id

    def revalue_open_positions(
        self,
        *,
        chain_snapshot: pd.DataFrame,
        fill_config: FillConfig | None = None,
    ) -> None:
        for pos in self.open_positions.values():
            repriced = reprice_matched_legs(
                matched_legs=pos.matched_legs,
                chain_snapshot=chain_snapshot,
                contract_multiplier=self.contract_multiplier,
                fill_config=fill_config,
            )
            if not repriced:
                continue

            if has_full_reprice(pos.matched_legs, repriced):
                pos.current_mark_value_cache = aggregate_repriced_mark_value(repriced)
                pos.current_exit_value_cache = aggregate_repriced_exit_value(repriced)

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

        leg_count = max(len(pos.matched_legs), 1)
        exit_commission = self.lifecycle_config.commission_per_contract * leg_count * pos.quantity
        exit_value = pos.exit_value() - (exit_commission / pos.quantity)
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
