from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

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

_EXCHANGE_TZ = ZoneInfo("America/New_York")


def live_chain_as_of_timestamp(
    now: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Exchange-calendar ``as_of`` date for live option DTE / expiry filters.

    Daily (and other lagged) primary bars often end on the prior session. Stamping
    Massive chains with that bar clock overstates DTE by the bar lag and admits
    expiries inside ``dte_min`` (e.g. true 6 DTE claimed as 7). Live matching
    must anchor to today's America/New_York date instead.
    """
    if now is None:
        et = pd.Timestamp.now(tz=_EXCHANGE_TZ)
    else:
        ts = pd.Timestamp(now)
        if ts.tzinfo is None:
            et = ts.tz_localize(_EXCHANGE_TZ)
        else:
            et = ts.tz_convert(_EXCHANGE_TZ)
    return pd.Timestamp(et.date())


def recompute_chain_dte(
    chain: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Rewrite ``dte`` as calendar days from *as_of* (default: live ET date)."""
    if chain is None or chain.empty or "expiry" not in chain.columns:
        return chain
    anchor = live_chain_as_of_timestamp() if as_of is None else pd.Timestamp(as_of).normalize()
    if getattr(anchor, "tzinfo", None) is not None:
        anchor = anchor.tz_localize(None)
    out = chain.copy()
    expiry = pd.to_datetime(out["expiry"])
    if getattr(expiry.dt, "tz", None) is not None:
        expiry = expiry.dt.tz_convert("UTC").dt.tz_localize(None)
    out["dte"] = (expiry.dt.normalize() - anchor).dt.days
    return out


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
    # Calendar-day DTE (expiry date − quote date). Massive stamps ``timestamp`` with
    # wall-clock "now", so ``(expiry - timestamp).dt.days`` undercounts by 1 for the
    # entire session after midnight and reports 0DTE as -1 — which drops same-day
    # expiries from ``dte_min=0`` windows and shifts three-track 7–21 bounds.
    ts = out["timestamp"]
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    out["dte"] = (out["expiry"].dt.normalize() - ts.dt.normalize()).dt.days

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
