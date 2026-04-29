"""Intraday 5-second microstructure volume-profile helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd

from rlm.data.microstructure.database.query import MicrostructureDB
from rlm.volume_profile.profile_calculator import calculate_volume_profile, identify_nodes


@lru_cache(maxsize=4096)
def _cached_profile(prices: tuple[float, ...], volumes: tuple[float, ...]) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2000-01-01", periods=len(prices), freq="S", tz="UTC"),
            "price": prices,
            "volume": volumes,
        }
    )
    profile = calculate_volume_profile(frame)
    nodes = identify_nodes(profile["volume_profile_series"])
    return {
        "poc": float(profile["poc"]),
        "value_area_high": float(profile["value_area_high"]),
        "value_area_low": float(profile["value_area_low"]),
        "hvn_levels": [float(x) for x in nodes["hvn_levels"]],
        "lvn_levels": [float(x) for x in nodes["lvn_levels"]],
    }


def _empty_profile() -> dict[str, Any]:
    return {
        "poc": float("nan"),
        "value_area_high": float("nan"),
        "value_area_low": float("nan"),
        "hvn_levels": [],
        "lvn_levels": [],
    }


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "price", "volume"])
    bars = df.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    price_col = "close" if "close" in bars.columns else "price"
    bars["price"] = pd.to_numeric(bars[price_col], errors="coerce")
    bars["volume"] = pd.to_numeric(bars.get("volume"), errors="coerce")
    return bars.dropna(subset=["timestamp", "price", "volume"])[["timestamp", "price", "volume"]]


def compute_intraday_vp(symbol: str, date: datetime, lookback_bars: int = 500) -> dict[str, Any]:
    """Compute intraday VP on the latest 5-second bars ending at ``date``."""
    end_ts = pd.Timestamp(date, tz="UTC") if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date).tz_convert("UTC")
    start_date = (end_ts - timedelta(days=2)).date().isoformat()
    end_date = end_ts.date().isoformat()

    db = MicrostructureDB()
    bars = db.load_underlying_bars(symbol.upper(), start_date, end_date, bar_resolution="5s")
    bars = _normalize_bars(bars)
    if bars.empty:
        return _empty_profile()

    bars = bars.loc[bars["timestamp"] <= end_ts].tail(int(lookback_bars))
    if bars.empty:
        return _empty_profile()

    prices = tuple(float(x) for x in bars["price"].to_numpy())
    volumes = tuple(float(x) for x in bars["volume"].to_numpy())
    return _cached_profile(prices, volumes)


def rolling_intraday_vp(symbol: str, window_seconds: int = 300) -> pd.DataFrame:
    """Compute rolling intraday VP metrics for each 5-second bar timestamp."""
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - timedelta(hours=8)

    db = MicrostructureDB()
    bars = db.load_underlying_bars(
        symbol.upper(), start_ts.date().isoformat(), end_ts.date().isoformat(), bar_resolution="5s"
    )
    bars = _normalize_bars(bars)
    if bars.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "vp_poc",
                "vp_va_high",
                "vp_va_low",
                "vp_hvn_count",
                "vp_lvn_count",
            ]
        )

    bars = bars.sort_values("timestamp").reset_index(drop=True)
    trailing = timedelta(seconds=int(window_seconds))
    out_rows: list[dict[str, Any]] = []

    for idx, row in bars.iterrows():
        window_start = row["timestamp"] - trailing
        window = bars.iloc[: idx + 1]
        window = window.loc[window["timestamp"] >= window_start]
        if window.empty:
            continue

        prices = tuple(float(x) for x in window["price"].to_numpy())
        volumes = tuple(float(x) for x in window["volume"].to_numpy())
        profile = _cached_profile(prices, volumes)
        out_rows.append(
            {
                "timestamp": row["timestamp"],
                "vp_poc": profile["poc"],
                "vp_va_high": profile["value_area_high"],
                "vp_va_low": profile["value_area_low"],
                "vp_hvn_count": len(profile["hvn_levels"]),
                "vp_lvn_count": len(profile["lvn_levels"]),
            }
        )

    return pd.DataFrame(out_rows)
