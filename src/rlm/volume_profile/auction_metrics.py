"""Auction-state metrics and profile trend analytics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def effort_result_divergence(df: pd.DataFrame, volume_profile: dict[str, Any]) -> float:
    """Return an effort-vs-result divergence score in ``[-1, 1]``.

    Positive values indicate bullish absorption (high effort with muted downside result);
    negative values indicate bearish absorption.
    """

    if df.empty:
        return 0.0

    work = df.copy()
    for col in ("high", "low", "open", "close", "volume"):
        if col not in work.columns:
            return 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["high", "low", "open", "close", "volume"])
    if work.empty:
        return 0.0

    candle_range = (work["high"] - work["low"]).clip(lower=0.0)
    range_pct_rank = candle_range.rank(pct=True).iloc[-1]

    vol_pct_rank = work["volume"].rank(pct=True).iloc[-1]
    directional_result = float(np.sign(work["close"].iloc[-1] - work["open"].iloc[-1]))

    imbalance = float(vol_pct_rank - range_pct_rank)
    score = directional_result * imbalance

    va_high = float(volume_profile.get("value_area_high", np.nan))
    va_low = float(volume_profile.get("value_area_low", np.nan))
    close = float(work["close"].iloc[-1])
    if np.isfinite(va_high) and close > va_high:
        score += 0.1
    elif np.isfinite(va_low) and close < va_low:
        score -= 0.1

    return float(np.clip(score, -1.0, 1.0))


def value_area_migration(profiles: list[dict[str, Any]]) -> str:
    """Classify value-area migration using the latest two profiles."""

    if len(profiles) < 2:
        return "neutral"

    prev = profiles[-2]
    curr = profiles[-1]

    prev_low = float(prev.get("value_area_low", np.nan))
    prev_high = float(prev.get("value_area_high", np.nan))
    curr_low = float(curr.get("value_area_low", np.nan))
    curr_high = float(curr.get("value_area_high", np.nan))

    if not all(np.isfinite([prev_low, prev_high, curr_low, curr_high])):
        return "neutral"

    prev_mid = (prev_low + prev_high) / 2.0
    curr_mid = (curr_low + curr_high) / 2.0

    overlaps = (curr_low <= prev_high) and (curr_high >= prev_low)
    if overlaps:
        return "neutral"
    if curr_mid > prev_mid:
        return "bullish"
    if curr_mid < prev_mid:
        return "bearish"
    return "neutral"


def auction_state(profile: dict[str, Any], price: float) -> str:
    """Classify a price versus value area."""

    va_low = float(profile.get("value_area_low", np.nan))
    va_high = float(profile.get("value_area_high", np.nan))
    if not np.isfinite(va_low) or not np.isfinite(va_high):
        return "balance"
    if price > va_high:
        return "imbalance_up"
    if price < va_low:
        return "imbalance_down"
    return "balance"
