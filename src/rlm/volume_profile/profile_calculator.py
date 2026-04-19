"""Core volume-profile calculations for auction-market analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"timestamp", "price", "volume"}


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for volume profile: {sorted(missing)}")

    out = df.copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["price", "volume"])
    if out.empty:
        raise ValueError("Input frame has no valid price/volume rows.")
    return out


def _value_area_bounds(volume_by_bin: pd.Series, value_area_percent: float) -> tuple[float, float]:
    if volume_by_bin.empty:
        return (np.nan, np.nan)

    target_volume = float(volume_by_bin.sum()) * float(value_area_percent)
    poc_idx = int(volume_by_bin.values.argmax())
    selected = {poc_idx}
    cumulative = float(volume_by_bin.iloc[poc_idx])
    left = poc_idx - 1
    right = poc_idx + 1

    while cumulative < target_volume and (left >= 0 or right < len(volume_by_bin)):
        left_vol = float(volume_by_bin.iloc[left]) if left >= 0 else -np.inf
        right_vol = float(volume_by_bin.iloc[right]) if right < len(volume_by_bin) else -np.inf

        if right_vol >= left_vol and right < len(volume_by_bin):
            selected.add(right)
            cumulative += right_vol
            right += 1
        elif left >= 0:
            selected.add(left)
            cumulative += left_vol
            left -= 1

    selected_prices = volume_by_bin.index[sorted(selected)]
    return float(selected_prices.min()), float(selected_prices.max())


def calculate_volume_profile(df: pd.DataFrame, price_precision: int = 400) -> dict[str, Any]:
    """Compute a volume profile and derived value-area levels.

    Parameters
    ----------
    df:
        Input data with ``timestamp``, ``price``, and ``volume`` columns.
    price_precision:
        Number of discretized price bins used to aggregate traded volume.

    Returns
    -------
    dict[str, Any]
        Dictionary containing POC, 70% VA bounds, 40% VA bounds, and the
        underlying volume profile series indexed by bin midpoint price.
    """

    clean = _validate_input(df)

    min_price = float(clean["price"].min())
    max_price = float(clean["price"].max())

    if np.isclose(min_price, max_price):
        index = pd.Index([min_price], dtype=float)
        profile = pd.Series([float(clean["volume"].sum())], index=index)
    else:
        bins = np.linspace(min_price, max_price, int(price_precision) + 1)
        bin_ids = pd.cut(clean["price"], bins=bins, include_lowest=True)
        profile = clean.groupby(bin_ids, observed=False)["volume"].sum()
        profile.index = profile.index.map(lambda iv: float((iv.left + iv.right) / 2.0))
        profile = profile.sort_index().astype(float)

    poc_price = float(profile.idxmax())
    va_low, va_high = _value_area_bounds(profile, value_area_percent=0.70)
    va40_low, va40_high = _value_area_bounds(profile, value_area_percent=0.40)

    return {
        "poc": poc_price,
        "value_area_high": va_high,
        "value_area_low": va_low,
        "value_area_40_high": va40_high,
        "value_area_40_low": va40_low,
        "volume_profile_series": profile,
    }


def identify_nodes(
    volume_profile_series: pd.Series, threshold_std: float = 1.0
) -> dict[str, list[float]]:
    """Identify high-volume and low-volume nodes from a profile series."""

    profile = volume_profile_series.dropna().astype(float)
    if profile.empty:
        return {"hvn_levels": [], "lvn_levels": []}

    mu = float(profile.mean())
    sigma = float(profile.std(ddof=0))

    hvn_mask = profile > (mu + threshold_std * sigma)
    lvn_mask = profile < (mu - threshold_std * sigma)

    return {
        "hvn_levels": [float(v) for v in profile.index[hvn_mask]],
        "lvn_levels": [float(v) for v in profile.index[lvn_mask]],
    }
