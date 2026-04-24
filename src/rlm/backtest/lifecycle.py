from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from rlm.backtest.commission import CommissionConfig
from rlm.backtest.cost_model import TransactionCostConfig


class ExpiryLiquidationPolicy(str, Enum):
    """Controls what happens to positions as they approach expiration.

    LIQUIDATE_BEFORE_EXPIRY:
        Force-close all positions when DTE reaches ``force_close_dte``.
        No position survives to expiry.
    SETTLE_AT_EXPIRY:
        Allow positions to run to expiry and settle them at intrinsic value.
        Positions are *not* force-closed before expiry by DTE.
    """

    LIQUIDATE_BEFORE_EXPIRY = "liquidate_before_expiry"
    SETTLE_AT_EXPIRY = "settle_at_expiry"


@dataclass(frozen=True)
class LifecycleConfig:
    force_close_dte: int = 1
    close_at_expiry_if_open: bool = True
    expiry_liquidation_policy: ExpiryLiquidationPolicy = (
        ExpiryLiquidationPolicy.LIQUIDATE_BEFORE_EXPIRY
    )
    max_holding_bars: int | None = None
    min_hold_bars: int = 2
    one_trade_per_bar: bool = True
    # Legacy scalar — kept for backward compatibility.  Prefer commission_config.
    commission_per_contract: float = 0.65
    commission_config: CommissionConfig = field(default_factory=CommissionConfig)
    transaction_cost_config: TransactionCostConfig = field(default_factory=TransactionCostConfig)

    def __post_init__(self) -> None:
        # Sync the legacy scalar with commission_config when it is still at the
        # default.  If the caller explicitly supplied commission_config, that
        # wins.  If only the legacy scalar differs from the default, build a
        # matching CommissionConfig so both sources agree.
        if (
            self.commission_config == CommissionConfig()
            and self.commission_per_contract != 0.65
        ):
            object.__setattr__(
                self,
                "commission_config",
                CommissionConfig.from_legacy_rate(self.commission_per_contract),
            )


def days_to_expiry(*, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None) -> int | None:
    if expiry is None:
        return None
    ts = pd.Timestamp(timestamp).normalize()
    exp = pd.Timestamp(expiry).normalize()
    return int((exp - ts).days)


def should_force_close_before_expiry(
    *, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None, config: LifecycleConfig
) -> bool:
    dte = days_to_expiry(timestamp=timestamp, expiry=expiry)
    return dte is not None and 0 < dte <= int(config.force_close_dte)


def is_at_or_past_expiry(*, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None) -> bool:
    dte = days_to_expiry(timestamp=timestamp, expiry=expiry)
    return dte is not None and dte <= 0


def should_close_for_max_holding(
    *, entry_bar_index: int | None, current_bar_index: int, config: LifecycleConfig
) -> bool:
    if config.max_holding_bars is None or entry_bar_index is None:
        return False
    return (current_bar_index - entry_bar_index) >= int(config.max_holding_bars)