"""Commission model for the backtesting engine.

Supports per-contract, per-trade (flat), and hybrid commission structures.
All dollar amounts are in the same currency unit as the portfolio cash.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommissionModel(str, Enum):
    """Supported commission calculation models."""

    PER_CONTRACT = "per_contract"
    """Charge a fixed amount per options contract traded."""

    PER_TRADE = "per_trade"
    """Charge a flat amount per trade (regardless of contract count)."""

    HYBRID = "hybrid"
    """Charge a flat per-trade base fee plus a per-contract fee."""


@dataclass(frozen=True)
class CommissionConfig:
    """Configuration for the commission model.

    Attributes
    ----------
    model:
        Which commission structure to apply.
    per_contract_rate:
        Dollar amount charged per contract (one leg = one contract).
        Used when ``model`` is :attr:`CommissionModel.PER_CONTRACT` or
        :attr:`CommissionModel.HYBRID`.
    per_trade_rate:
        Flat dollar amount charged once per trade (open or close event).
        Used when ``model`` is :attr:`CommissionModel.PER_TRADE` or
        :attr:`CommissionModel.HYBRID`.
    min_commission:
        Floor applied to the total commission for a single trade event.
        Set to ``0.0`` to disable.
    """

    model: CommissionModel = CommissionModel.PER_CONTRACT
    per_contract_rate: float = 0.65
    per_trade_rate: float = 0.0
    min_commission: float = 0.0

    @classmethod
    def from_legacy_rate(cls, commission_per_contract: float) -> "CommissionConfig":
        """Build a :class:`CommissionConfig` from a simple per-contract rate.

        Convenience factory to migrate code that previously used the old
        ``LifecycleConfig.commission_per_contract`` scalar.
        """
        return cls(
            model=CommissionModel.PER_CONTRACT,
            per_contract_rate=commission_per_contract,
        )


def calculate_commission(
    *,
    config: CommissionConfig,
    leg_count: int,
    quantity: int,
) -> float:
    """Compute the total commission for a single trade event (open or close).

    Parameters
    ----------
    config:
        The :class:`CommissionConfig` to apply.
    leg_count:
        Number of option legs in the trade.
    quantity:
        Number of contracts per leg (i.e. the position quantity multiplier).

    Returns
    -------
    float
        Total commission in dollars (always >= 0).
    """
    total_contracts = leg_count * quantity

    if config.model == CommissionModel.PER_CONTRACT:
        commission = config.per_contract_rate * total_contracts
    elif config.model == CommissionModel.PER_TRADE:
        commission = config.per_trade_rate
    elif config.model == CommissionModel.HYBRID:
        commission = config.per_trade_rate + config.per_contract_rate * total_contracts
    else:
        commission = config.per_contract_rate * total_contracts

    return max(float(commission), float(config.min_commission))
