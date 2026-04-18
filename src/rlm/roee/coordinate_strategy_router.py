from __future__ import annotations

from typing import Mapping


def select_strategy_from_coordinates(row: Mapping[str, str]) -> str:
    regime = row["M_regime"]
    if regime == "trend_up_stable":
        return "bull_call_spread"
    if regime == "trend_down_stable":
        return "bear_put_spread"
    if regime == "range_compression":
        return "iron_condor"
    if regime == "mean_reversion":
        return "iron_condor"
    if regime == "breakout_expansion":
        return "debit_spread"
    if regime == "transition":
        return "calendar_spread"
    return "no_trade"
