"""
Implied Volatility (IV) surface builder for the RLM microstructure layer.

Builds a smooth, queryable vol surface from raw option chain snapshots using
``scipy.interpolate.griddata`` (cubic interpolation on a moneyness × DTE grid).

The surface is stored in the standard RBF ("radial basis") format so downstream
callers can quickly retrieve the interpolated IV at any (moneyness, DTE) point
without re-computing from scratch.

Usage::

    from rlm.microstructure.calculators.iv_surface import (
        build_iv_surface,
        query_iv_surface,
        save_iv_surface,
    )

    # Build from a greeks snapshot DataFrame
    surface = build_iv_surface(snapshot_df, timestamp="2025-06-10 15:30:00", underlying_symbol="SPY")

    # Query a point
    iv_at_atm_30dte = query_iv_surface(surface, moneyness=1.0, dte=30)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata


# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

_MONEYNESS_GRID = np.linspace(0.50, 1.50, 60)    # K/S from deep ITM to deep OTM
_DTE_GRID = np.array([                             # Key DTE breakpoints (non-linear)
    1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365
], dtype=float)


# ---------------------------------------------------------------------------
# Surface construction
# ---------------------------------------------------------------------------

def build_iv_surface(
    snapshot: pd.DataFrame,
    *,
    timestamp: "str | pd.Timestamp",
    underlying_symbol: str,
    method: Literal["cubic", "linear", "nearest"] = "cubic",
    min_points: int = 10,
) -> pd.DataFrame:
    """
    Interpolate a smooth IV surface from a snapshot DataFrame.

    Parameters
    ----------
    snapshot            : DataFrame with columns:
                          implied_vol, moneyness (or strike + underlying_price),
                          dte (days to expiry)
    timestamp           : Snapshot timestamp
    underlying_symbol   : Ticker (e.g. "SPY")
    method              : scipy griddata interpolation method
    min_points          : Minimum valid (non-NaN) contracts required

    Returns
    -------
    DataFrame with columns:
        timestamp, underlying_symbol, moneyness, days_to_expiry, implied_vol, raw_iv_count
    """
    df = snapshot.copy()

    # Derive moneyness if not present
    if "moneyness" not in df.columns:
        if "strike" not in df.columns or "underlying_price" not in df.columns:
            raise ValueError("snapshot must have 'moneyness' or ('strike', 'underlying_price')")
        df["moneyness"] = df["strike"] / df["underlying_price"]

    # Derive DTE if not present
    if "dte" not in df.columns:
        if "expiration" not in df.columns:
            raise ValueError("snapshot must have 'dte' or 'expiration' column")
        ref_date = pd.Timestamp(timestamp)
        df["dte"] = (pd.to_datetime(df["expiration"]) - ref_date).dt.days.clip(lower=0)

    # Clean: drop NaN IV, extreme moneyness / DTE, and zero IV
    df = df.dropna(subset=["implied_vol", "moneyness", "dte"])
    df = df[
        (df["implied_vol"] > 0.01) & (df["implied_vol"] < 5.0)  # 1%-500% IV range
        & (df["moneyness"] >= 0.50) & (df["moneyness"] <= 1.50)
        & (df["dte"] >= 1) & (df["dte"] <= 365)
    ]

    raw_count = len(df)
    if raw_count < min_points:
        warnings.warn(
            f"iv_surface: only {raw_count} valid contracts for {underlying_symbol} "
            f"@ {timestamp}; surface will be sparse.",
            stacklevel=2,
        )

    # Interpolate
    if raw_count >= min_points:
        points = df[["moneyness", "dte"]].values
        values = df["implied_vol"].values
        grid_m, grid_d = np.meshgrid(_MONEYNESS_GRID, _DTE_GRID)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_iv = griddata(points, values, (grid_m, grid_d), method=method)
    else:
        grid_m, grid_d = np.meshgrid(_MONEYNESS_GRID, _DTE_GRID)
        grid_iv = np.full(grid_m.shape, float("nan"))

    surface = pd.DataFrame({
        "timestamp": str(timestamp),
        "underlying_symbol": underlying_symbol,
        "moneyness": grid_m.ravel(),
        "days_to_expiry": grid_d.ravel(),
        "implied_vol": grid_iv.ravel(),
        "raw_iv_count": raw_count,
    }).dropna(subset=["implied_vol"])

    return surface.reset_index(drop=True)


def build_iv_surface_from_parquet(
    conn: "duckdb.DuckDBPyConnection",
    *,
    symbol: str,
    timestamp: "str | pd.Timestamp",
    data_path: str = "data/microstructure",
    method: Literal["cubic", "linear", "nearest"] = "cubic",
) -> pd.DataFrame:
    """
    Build IV surface by querying greeks snapshots Parquet via DuckDB.
    """
    parquet_glob = f"{data_path}/options/{symbol}/greeks_snapshots/*.parquet"
    query = f"""
        SELECT
            implied_vol,
            strike,
            underlying_price,
            dte,
            expiration
        FROM '{parquet_glob}'
        WHERE underlying_symbol = '{symbol}'
          AND timestamp = TIMESTAMP '{timestamp}'
          AND implied_vol IS NOT NULL
    """
    try:
        snapshot = conn.execute(query).fetchdf()
    except Exception as exc:
        raise RuntimeError(f"DuckDB query failed: {exc}") from exc

    if snapshot.empty:
        return pd.DataFrame()

    return build_iv_surface(snapshot, timestamp=timestamp, underlying_symbol=symbol, method=method)


# ---------------------------------------------------------------------------
# Surface query
# ---------------------------------------------------------------------------

def query_iv_surface(
    surface: pd.DataFrame,
    *,
    moneyness: float,
    dte: float,
) -> float:
    """
    Retrieve the interpolated IV from a pre-built surface DataFrame.

    Uses nearest-neighbour lookup on the grid (the grid is already dense enough
    for practical use).  Returns NaN if the surface is empty.
    """
    if surface.empty or "implied_vol" not in surface.columns:
        return float("nan")

    # Squared distance on normalised axes
    m_range = _MONEYNESS_GRID[-1] - _MONEYNESS_GRID[0]
    d_range = float(_DTE_GRID[-1] - _DTE_GRID[0])

    dm = (surface["moneyness"] - moneyness) / m_range
    dd = (surface["days_to_expiry"] - dte) / d_range
    dist = dm ** 2 + dd ** 2

    idx = dist.idxmin()
    return float(surface.loc[idx, "implied_vol"])


def skew_at_dte(surface: pd.DataFrame, dte: float, *, delta_otm: float = 0.10) -> float:
    """
    Return the 25-delta put-call skew: IV(K/S = 1 − delta_otm) − IV(K/S = 1 + delta_otm).

    Positive skew = puts more expensive than equidistant calls (typical for equity).
    """
    iv_put = query_iv_surface(surface, moneyness=1.0 - delta_otm, dte=dte)
    iv_call = query_iv_surface(surface, moneyness=1.0 + delta_otm, dte=dte)
    return iv_put - iv_call


def term_structure(surface: pd.DataFrame, moneyness: float = 1.0) -> pd.Series:
    """
    Return the ATM term structure (IV vs DTE) for a given moneyness level.
    """
    out = {}
    for dte in _DTE_GRID:
        out[dte] = query_iv_surface(surface, moneyness=moneyness, dte=float(dte))
    return pd.Series(out, name="atm_iv")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_iv_surface(surface: pd.DataFrame, symbol: str, data_path: str = "data/microstructure") -> None:
    """Persist IV surface to date-partitioned Parquet."""
    if surface.empty:
        return

    df = surface.copy()
    df["_date"] = pd.to_datetime(df["timestamp"]).dt.date

    for date, group in df.groupby("_date"):
        path = Path(data_path) / f"options/{symbol}/derived/iv_surface"
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{symbol}_{date}.parquet"
        group.drop(columns="_date").to_parquet(file_path, index=False)
