from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_CHAIN_COLUMNS = {
    "timestamp",
    "underlying",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
}


@dataclass(frozen=True)
class ChainFilter:
    underlying: str | None = None
    timestamp: pd.Timestamp | None = None
    expiry_min_days: int | None = None
    expiry_max_days: int | None = None


def validate_option_chain(df: pd.DataFrame) -> None:
    missing = REQUIRED_CHAIN_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Option chain missing required columns: {sorted(missing)}")


def option_chain_is_usable(df: pd.DataFrame | None) -> bool:
    """True if *df* has all :data:`REQUIRED_CHAIN_COLUMNS` (may be empty rows)."""
    if df is None or df.empty:
        return False
    return REQUIRED_CHAIN_COLUMNS.issubset(df.columns)


def normalize_option_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalized schema:
      timestamp, underlying, expiry, option_type, strike, bid, ask, mid, dte,
      delta?, gamma?, theta?, vega?, iv?, open_interest?, volume?
    """
    validate_option_chain(df)

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["option_type"] = out["option_type"].str.lower().str.strip()
    out["strike"] = out["strike"].astype(float)
    out["bid"] = out["bid"].astype(float)
    out["ask"] = out["ask"].astype(float)

    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]
    out["spread_pct_mid"] = np.where(out["mid"] > 0, out["spread"] / out["mid"], np.nan)
    out["dte"] = (out["expiry"] - out["timestamp"]).dt.days

    numeric_optional = [
        "delta",
        "gamma",
        "theta",
        "vega",
        "iv",
        "iv_greeks",
        "rho",
        "charm",
        "vanna",
        "open_interest",
        "volume",
    ]
    for col in numeric_optional:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def filter_option_chain(
    df: pd.DataFrame,
    chain_filter: ChainFilter | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if chain_filter is None:
        return out

    if chain_filter.underlying is not None:
        out = out[out["underlying"] == chain_filter.underlying]

    if chain_filter.timestamp is not None:
        ts = pd.Timestamp(chain_filter.timestamp)
        out = out[out["timestamp"] == ts]

    if chain_filter.expiry_min_days is not None:
        out = out[out["dte"] >= chain_filter.expiry_min_days]

    if chain_filter.expiry_max_days is not None:
        out = out[out["dte"] <= chain_filter.expiry_max_days]

    return out


def calculate_dte_from_expiry(
    expiry: pd.Timestamp | str,
    timestamp: pd.Timestamp | str,
) -> float:
    """Calendar days from timestamp to expiry, consistent with normalize_option_chain."""
    return float((pd.Timestamp(expiry).normalize() - pd.Timestamp(timestamp).normalize()).days)


def select_nearest_expiry_slice(
    chain: pd.DataFrame,
    dte_min: int,
    dte_max: int,
) -> pd.DataFrame:
    eligible = chain[(chain["dte"] >= dte_min) & (chain["dte"] <= dte_max)].copy()
    if eligible.empty:
        return eligible

    target_dte = (dte_min + dte_max) / 2.0
    expiries = (
        eligible[["expiry", "dte"]]
        .drop_duplicates()
        .assign(dte_distance=lambda x: (x["dte"] - target_dte).abs())
        .sort_values(["dte_distance", "dte", "expiry"])
    )

    best_expiry = expiries.iloc[0]["expiry"]
    return eligible[eligible["expiry"] == best_expiry].copy()
