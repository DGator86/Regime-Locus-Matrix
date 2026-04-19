"""Volume-profile based trade models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rlm.volume_profile.profile_calculator import calculate_volume_profile


def eighty_percent_rule(current_open: float, prev_profile: dict[str, Any]) -> dict[str, Any]:
    """Generate a simplified 80% rule signal from previous-session value area."""

    va_low = float(prev_profile.get("value_area_low", np.nan))
    va_high = float(prev_profile.get("value_area_high", np.nan))
    poc = float(prev_profile.get("poc", np.nan))

    if not np.isfinite(va_low) or not np.isfinite(va_high):
        return {"signal": False, "direction": None, "target": np.nan, "stop": np.nan}

    if current_open < va_low:
        # Re-entry trigger proxied by being between low and POC.
        if np.isfinite(poc) and current_open >= (va_low - abs(va_low - poc) * 0.2):
            return {"signal": True, "direction": "long", "target": va_high, "stop": va_low}
    elif current_open > va_high:
        if np.isfinite(poc) and current_open <= (va_high + abs(va_high - poc) * 0.2):
            return {"signal": True, "direction": "short", "target": va_low, "stop": va_high}

    return {"signal": False, "direction": None, "target": np.nan, "stop": np.nan}


def core_value_supply_demand(
    df: pd.DataFrame, lookback: int = 50
) -> dict[str, list[dict[str, float]]]:
    """Build simple supply/demand zones using swing pivots and 40% value area anchors."""

    if df.empty or not {"high", "low", "close", "volume"}.issubset(df.columns):
        return {"demand_zones": [], "supply_zones": []}

    work = df.tail(lookback).copy()
    for col in ("high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["high", "low", "close", "volume"])
    if len(work) < 5:
        return {"demand_zones": [], "supply_zones": []}

    work["price"] = work["close"]
    work["timestamp"] = (
        pd.to_datetime(work.index, utc=True, errors="coerce")
        if "timestamp" not in work.columns
        else pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    )
    profile = calculate_volume_profile(work[["timestamp", "price", "volume"]])

    va40_low = float(profile.get("value_area_40_low", np.nan))
    va40_high = float(profile.get("value_area_40_high", np.nan))

    swing_high = float(work["high"].rolling(3, center=True).max().max())
    swing_low = float(work["low"].rolling(3, center=True).min().min())

    demand = []
    supply = []
    if np.isfinite(va40_low):
        demand.append(
            {
                "low": min(swing_low, va40_low),
                "high": max(swing_low, va40_low),
            }
        )
    if np.isfinite(va40_high):
        supply.append(
            {
                "low": min(swing_high, va40_high),
                "high": max(swing_high, va40_high),
            }
        )

    return {"demand_zones": demand, "supply_zones": supply}


def institutional_fair_value(
    rth_profile: dict[str, Any],
    eth_profile: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """Assess likely institutional reaction around RTH/ETH value references."""

    zones: list[tuple[str, float]] = []
    for name, profile in (("rth", rth_profile), ("eth", eth_profile)):
        poc = float(profile.get("poc", np.nan))
        va_low = float(profile.get("value_area_low", np.nan))
        va_high = float(profile.get("value_area_high", np.nan))
        for zone in (poc, va_low, va_high):
            if np.isfinite(zone):
                zones.append((name, zone))

    if not zones:
        return {"reaction_expected": False, "zone_type": "demand", "confidence": 0.0}

    nearest = min(zones, key=lambda item: abs(item[1] - current_price))
    distance = abs(nearest[1] - current_price)
    conf = float(np.clip(1.0 - (distance / max(abs(current_price), 1e-6)), 0.0, 1.0))
    zone_type = "demand" if current_price <= nearest[1] else "supply"
    return {
        "reaction_expected": conf >= 0.95,
        "zone_type": zone_type,
        "confidence": conf,
    }
