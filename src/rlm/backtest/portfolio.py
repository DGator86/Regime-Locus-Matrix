from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rlm.backtest.commission import calculate_commission
from rlm.backtest.cost_model import calculate_transaction_cost
from rlm.backtest.expiry import settle_legs_at_expiry
from rlm.backtest.fills import (
    FillConfig,
    entry_fill_price,
    exit_fill_price,
    signed_cashflow_for_fill,
)
from rlm.backtest.lifecycle import LifecycleConfig
from rlm.backtest.revalue import (
    aggregate_repriced_exit_value,
    aggregate_repriced_mark_value,
    reprice_matched_legs_detailed,
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
        """Calculate PnL percentage. Long options are capped at -100% (plus fees)."""
        denom = abs(self.entry_cost) if abs(self.entry_cost) > 1e-9 else np.nan
        if not pd.notna(denom):
            return np.nan

        raw_pct = self.pnl() / denom

        # For long positions (debit > 0), market loss cannot exceed 100%.
        # entry_cost includes fees, so pnl can be slightly more than -100%.
        # We cap it to avoid -400% artifacts.
        is_long = self.entry_cost > 0
        if is_long:
            return max(raw_pct, -1.05)  # Allow 5% for fees/slippage

        return raw_pct


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

    def has_position_for_symbol(self, symbol: str) -> bool:
        """Returns True if there is an open position for the given symbol."""
        return any(pos.underlying_symbol == symbol for pos in self.open_positions.values())

    def total_position_count(self) -> int:
        """Returns the number of open positions."""
        return len(self.open_positions)

    def can_open(
        self,
        net_premium_debit_per_unit: float,
        entry_friction: float,
        quantity: int,
        risk_capital: float,
    ) -> bool:
        """Cash needed to open: friction + max(premium outflow, risk capital).

        ``net_premium_debit_per_unit`` is the signed net from :meth:`_compute_entry_cost` (positive = pay debit).
        """
        total_premium_flow = float(net_premium_debit_per_unit) * int(quantity)
        premium_cash_need = max(0.0, total_premium_flow)
        required = float(entry_friction) + max(premium_cash_need, float(risk_capital))
        return self.cash >= required

    def _compute_entry_cost(
        self,
        *,
        matched_legs: list[dict],
        fill_config: FillConfig | None = None,
        quantity: int = 1,
    ) -> float:
        cfg = fill_config or FillConfig(contract_multiplier=self.contract_multiplier)

        total = 0.0
        for leg in matched_legs:
            premium = entry_fill_price(
                side=str(leg["side"]),
                bid=float(leg["bid"]),
                ask=float(leg["ask"]),
                config=cfg,
                quantity=quantity,
                quote_size=float(leg.get("ask_size") or leg.get("bid_size") or 0.0) or None,
            )
            total += signed_cashflow_for_fill(
                side=str(leg["side"]),
                premium=premium,
                contract_multiplier=self.contract_multiplier,
            )
        return float(total)

    @staticmethod
    def _capital_reserve_for_structure(
        matched_legs: list[dict],
        contract_multiplier: int,
        quantity: int,
    ) -> float:
        """Conservative buying-power reserve for defined-risk multi-leg structures (strike span × multiplier)."""
        if len(matched_legs) < 2:
            return 0.0
        strikes = [float(leg["strike"]) for leg in matched_legs]
        width = max(strikes) - min(strikes)
        if width <= 0.0:
            return 0.0
        return float(width) * float(contract_multiplier) * float(quantity)

    @classmethod
    def _risk_capital_per_unit(
        cls,
        *,
        matched_legs: list[dict],
        contract_multiplier: int,
        entry_cost: float,
    ) -> float:
        reserve = cls._capital_reserve_for_structure(
            matched_legs=matched_legs,
            contract_multiplier=contract_multiplier,
            quantity=1,
        )
        if entry_cost >= 0.0:
            return float(max(entry_cost, 0.0))
        if reserve > 0.0:
            return float(max(reserve + entry_cost, 0.0))
        return float(abs(entry_cost))

    def _initial_marks_from_matched_legs(
        self,
        *,
        matched_legs: list[dict],
        fill_config: FillConfig,
    ) -> tuple[float, float]:
        """Mid mark and executable exit value (per unit, × multiplier) right after entry fills."""
        mark_total = 0.0
        exit_total = 0.0
        for leg in matched_legs:
            side = str(leg["side"])
            bid = float(leg["bid"])
            ask = float(leg["ask"])
            mid = float(leg.get("mid", (bid + ask) / 2.0))
            signed_mid = mid if side == "long" else -mid
            mark_total += signed_mid * self.contract_multiplier
            exe = exit_fill_price(
                side=side,
                bid=bid,
                ask=ask,
                config=fill_config,
                quote_size=float(leg.get("bid_size") or leg.get("ask_size") or 0.0) or None,
            )
            signed_ex = exe if side == "long" else -exe
            exit_total += signed_ex * self.contract_multiplier
        return float(mark_total), float(exit_total)

    def _resolve_position_quantity(
        self,
        *,
        decision: TradeDecision,
        matched_legs: list[dict],
        underlying_price: float,
        base_quantity: int,
        fill_config: FillConfig,
    ) -> int:
        min_quantity = max(int(base_quantity), 1)
        if decision.size_fraction is None:
            return min_quantity
        size_fraction = float(decision.size_fraction or 0.0)
        if size_fraction <= 0.0:
            return 0

        entry_cost = self._compute_entry_cost(matched_legs=matched_legs, fill_config=fill_config, quantity=quantity)
        entry_friction_per_unit = (
            calculate_commission(
                config=self.lifecycle_config.commission_config,
                leg_count=max(len(matched_legs), 1),
                quantity=1,
            )
            + calculate_transaction_cost(
                matched_legs=matched_legs,
                underlying_price=underlying_price,
                quantity=1,
                contract_multiplier=self.contract_multiplier,
                config=self.lifecycle_config.transaction_cost_config,
            ).total
        )
        risk_capital_per_unit = self._risk_capital_per_unit(
            matched_legs=matched_legs,
            contract_multiplier=self.contract_multiplier,
            entry_cost=entry_cost,
        )
        capital_per_unit = max(max(entry_cost, 0.0), risk_capital_per_unit) + entry_friction_per_unit
        if capital_per_unit <= 1e-9:
            return min_quantity

        budget = self.equity() * size_fraction
        sized_quantity = int(budget // capital_per_unit)
        if sized_quantity <= 0:
            return 0
        return max(min_quantity, sized_quantity)

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

        cfg = fill_config or FillConfig(contract_multiplier=self.contract_multiplier)
        actual_quantity = self._resolve_position_quantity(
            decision=decision,
            matched_legs=matched,
            underlying_price=underlying_price,
            base_quantity=quantity,
            fill_config=cfg,
        )
        if actual_quantity <= 0:
            return None

        entry_cost = self._compute_entry_cost(matched_legs=matched, fill_config=cfg, quantity=1)
        leg_count = max(len(matched), 1)
        entry_commission = calculate_commission(
            config=self.lifecycle_config.commission_config,
            leg_count=leg_count,
            quantity=actual_quantity,
        )
        entry_transaction_cost = calculate_transaction_cost(
            matched_legs=matched,
            underlying_price=underlying_price,
            quantity=actual_quantity,
            contract_multiplier=self.contract_multiplier,
            config=self.lifecycle_config.transaction_cost_config,
        )
        total_entry_cost = entry_cost * actual_quantity
        risk_capital = (
            self._risk_capital_per_unit(
                matched_legs=matched,
                contract_multiplier=self.contract_multiplier,
                entry_cost=entry_cost,
            )
            * actual_quantity
        )

        if not self.can_open(
            entry_cost,
            entry_commission + entry_transaction_cost.total,
            actual_quantity,
            risk_capital,
        ):
            return None

        self.cash -= total_entry_cost
        self.cash -= entry_commission
        self.cash -= entry_transaction_cost.total

        expiry = matched[0]["expiry"] if matched else None
        position_id = str(uuid.uuid4())
        init_mark, init_exit = self._initial_marks_from_matched_legs(matched_legs=matched, fill_config=cfg)
        meta = dict(decision.metadata)
        meta["reprice_ok"] = True
        meta["last_reprice"] = {"full": True, "missing_leg_count": 0, "stale": False}
        meta["requested_size_fraction"] = float(decision.size_fraction or 0.0)
        meta["resolved_quantity"] = int(actual_quantity)
        meta["risk_capital"] = float(risk_capital)
        meta["entry_cost_breakdown"] = {
            "premium_cashflow": float(total_entry_cost),
            "commission": float(entry_commission),
            **entry_transaction_cost.to_dict(),
        }

        pos = OpenPosition(
            position_id=position_id,
            timestamp_entry=pd.Timestamp(timestamp),
            strategy_name=decision.strategy_name or "unknown",
            regime_key=decision.regime_key or "unknown",
            underlying_symbol=underlying_symbol,
            entry_underlying_price=float(underlying_price),
            expiry=expiry,
            entry_cost=float(entry_cost + ((entry_commission + entry_transaction_cost.total) / actual_quantity)),
            quantity=int(actual_quantity),
            matched_legs=matched,
            target_profit_pct=float(decision.target_profit_pct or 0.5),
            max_risk_pct=float(decision.max_risk_pct or 0.02),
            metadata=meta,
            entry_bar_index=bar_index,
        )

        pos.current_mark_value_cache = init_mark
        pos.current_exit_value_cache = init_exit

        self.open_positions[position_id] = pos
        return position_id

    def revalue_open_positions(
        self,
        *,
        chain_snapshot: pd.DataFrame,
        fill_config: FillConfig | None = None,
    ) -> None:
        for pos in self.open_positions.values():
            result = reprice_matched_legs_detailed(
                matched_legs=pos.matched_legs,
                chain_snapshot=chain_snapshot,
                contract_multiplier=self.contract_multiplier,
                fill_config=fill_config,
            )
            if result.is_full:
                pos.current_mark_value_cache = aggregate_repriced_mark_value(result.legs)
                pos.current_exit_value_cache = aggregate_repriced_exit_value(result.legs)
                pos.metadata["reprice_ok"] = True
                pos.metadata["last_reprice"] = {
                    "full": True,
                    "missing_leg_count": 0,
                    "stale": False,
                }
            elif result.legs:
                pos.metadata["reprice_ok"] = False
                pos.metadata["last_reprice"] = {
                    "full": False,
                    "missing_leg_count": result.missing_leg_count,
                    "stale": True,
                }
            else:
                pos.metadata["reprice_ok"] = pos.metadata.get("reprice_ok", True)
                pos.metadata["last_reprice"] = {
                    "full": False,
                    "missing_leg_count": result.missing_leg_count,
                    "stale": True,
                }

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
        exit_commission = calculate_commission(
            config=self.lifecycle_config.commission_config,
            leg_count=leg_count,
            quantity=pos.quantity,
        )
        exit_transaction_cost = calculate_transaction_cost(
            matched_legs=pos.matched_legs,
            underlying_price=underlying_price,
            quantity=pos.quantity,
            contract_multiplier=self.contract_multiplier,
            config=self.lifecycle_config.transaction_cost_config,
        )
        exit_value = pos.exit_value() - ((exit_commission + exit_transaction_cost.total) / pos.quantity)
        total_exit_value = exit_value * pos.quantity
        self.cash += total_exit_value

        pnl = (exit_value - pos.entry_cost) * pos.quantity
        pnl_pct = pnl / (abs(pos.entry_cost) * pos.quantity) if abs(pos.entry_cost) > 1e-9 else np.nan
        metadata = dict(pos.metadata)
        metadata["exit_cost_breakdown"] = {
            "commission": float(exit_commission),
            **exit_transaction_cost.to_dict(),
        }

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
            metadata=metadata,
        )

        self.closed_trades.append(trade)
        del self.open_positions[position_id]
        return trade

    def expiry_settle_position(
        self,
        *,
        position_id: str,
        timestamp_exit: pd.Timestamp,
        underlying_price: float,
    ) -> TradeRecord | None:
        """Settle a position at expiry using intrinsic value.

        Unlike :meth:`close_position` (which uses the cached mark/exit
        values from market-data repricing), this method computes payoff
        directly from the legs' intrinsic values at the settlement price.
        No closing commissions are charged — expiry settlement is assumed
        to be commission-free (assignment fees can be extended here if
        needed in the future).

        The portfolio cash is credited or debited by the net settlement
        amount.  For short ITM legs the assignment is simulated and the
        cash impact is correctly negative.
        """
        pos = self.open_positions.get(position_id)
        if pos is None:
            return None

        settlement = settle_legs_at_expiry(
            legs=pos.matched_legs,
            underlying_price=float(underlying_price),
            contract_multiplier=self.contract_multiplier,
        )

        # Cash impact per-unit * quantity
        total_cash_impact = settlement.cash_impact * pos.quantity
        self.cash += total_cash_impact

        # Normalize exit_value to per-unit for the trade record
        exit_value_per_unit = settlement.intrinsic_value * self.contract_multiplier
        pnl = (exit_value_per_unit - pos.entry_cost) * pos.quantity
        pnl_pct = pnl / (abs(pos.entry_cost) * pos.quantity) if abs(pos.entry_cost) > 1e-9 else np.nan

        metadata = dict(pos.metadata)
        metadata["settlement_notes"] = settlement.notes
        metadata["assignment_occurred"] = settlement.assignment_occurred

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
            exit_value=exit_value_per_unit * pos.quantity,
            pnl=float(pnl),
            pnl_pct=float(pnl_pct) if pd.notna(pnl_pct) else np.nan,
            quantity=pos.quantity,
            exit_reason="expiry_settlement",
            metadata=metadata,
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
