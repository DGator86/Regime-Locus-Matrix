from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LifecycleConfig:
    force_close_dte: int = 1
    close_at_expiry_if_open: bool = True
    max_holding_bars: int | None = None
    one_trade_per_bar: bool = True
    commission_per_contract: float = 0.65


def days_to_expiry(*, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None) -> int | None:
    if expiry is None:
        return None
    ts = pd.Timestamp(timestamp).normalize()
    exp = pd.Timestamp(expiry).normalize()
    return int((exp - ts).days)


def should_force_close_before_expiry(*, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None, config: LifecycleConfig) -> bool:
    dte = days_to_expiry(timestamp=timestamp, expiry=expiry)
    return dte is not None and 0 < dte <= int(config.force_close_dte)


def is_at_or_past_expiry(*, timestamp: pd.Timestamp, expiry: str | pd.Timestamp | None) -> bool:
    dte = days_to_expiry(timestamp=timestamp, expiry=expiry)
    return dte is not None and dte <= 0


def should_close_for_max_holding(*, entry_bar_index: int | None, current_bar_index: int, config: LifecycleConfig) -> bool:
    if config.max_holding_bars is None or entry_bar_index is None:
        return False
    return (current_bar_index - entry_bar_index) >= int(config.max_holding_bars)
