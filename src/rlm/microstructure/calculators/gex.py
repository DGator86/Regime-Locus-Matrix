"""
Gamma Exposure (GEX) surface builder for the RLM microstructure layer.

Conventions (SqueezeMetrics / SpotGamma standard)
--------------------------------------------------
  GEX per contract  = gamma × open_interest × 100 × spot

  - Calls are dealer-LONG (positive GEX → market-maker hedging stabilises price)
  - Puts are dealer-SHORT  (negative GEX → market-maker hedging amplifies moves)
  - net_gex = call_gex − |put_gex|

When net_gex > 0 the market-maker community is net-long gamma and acts as a
shock-absorber; when net_gex < 0 they are net-short gamma and can accelerate
moves.  The GEX flip level (where net_gex crosses zero as a function of spot)
is the key "regime locus" that this module is designed to expose.

Usage::

    from rlm.microstructure.calculators.gex import build_gex_surface, gex_flip_level
    import duckdb

    conn = duckdb.connect()
    gex_df = build_gex_surface(conn, symbol="SPY", timestamp="2025-06-10 15:30:00")
    flip  = gex_flip_level(gex_df)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# In-memory GEX surface (from a greeks snapshot DataFrame)
# ---------------------------------------------------------------------------

def build_gex_surface_from_df(
    snapshot: pd.DataFrame,
    *,
    underlying_symbol: str,
    timestamp: "str | pd.Timestamp",
) -> pd.DataFrame:
    """
    Build the GEX surface from a greeks snapshot DataFrame.

    Parameters
    ----------
    snapshot            : DataFrame with columns: strike, expiration, option_type,
                          gamma, open_interest, underlying_price
    underlying_symbol   : Ticker string (e.g. "SPY")
    timestamp           : Timestamp for this snapshot

    Returns
    -------
    DataFrame with columns:
        timestamp, underlying_symbol, underlying_price, strike, expiration,
        total_gamma, call_gex, put_gex, net_gex, gex, gex_pct
    """
    required = {"strike", "expiration", "option_type", "gamma", "open_interest", "underlying_price"}
    missing = required - set(snapshot.columns)
    if missing:
        raise ValueError(f"snapshot missing columns: {missing}")

    df = snapshot.copy()
    df["option_type"] = df["option_type"].str.lower()

    spot = float(df["underlying_price"].iloc[0]) if "underlying_price" in df.columns else float("nan")

    # Per-contract GEX
    df["_gex_raw"] = df["gamma"] * df["open_interest"] * 100.0 * spot
    df["_is_call"] = df["option_type"] == "call"

    grouped = df.groupby(["strike", "expiration"], sort=True)

    rows = []
    for (strike, expiration), grp in grouped:
        calls = grp[grp["_is_call"]]
        puts = grp[~grp["_is_call"]]

        call_gex = float(calls["_gex_raw"].sum())
        put_gex = -float(puts["_gex_raw"].sum())  # Negative: dealers are short puts
        total_gamma = float(grp["gamma"].sum())
        gex = call_gex + put_gex  # = net dealer GEX
        rows.append({
            "timestamp": timestamp,
            "underlying_symbol": underlying_symbol,
            "underlying_price": spot,
            "strike": float(strike),
            "expiration": expiration,
            "total_gamma": total_gamma,
            "call_gex": call_gex,
            "put_gex": put_gex,
            "net_gex": call_gex - abs(put_gex),
            "gex": gex,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    total_abs_gex = result["gex"].abs().sum()
    result["gex_pct"] = result["gex"] / total_abs_gex if total_abs_gex > 0 else 0.0
    return result.sort_values("strike").reset_index(drop=True)


def build_gex_surface(
    conn: "duckdb.DuckDBPyConnection",
    *,
    symbol: str,
    timestamp: "str | pd.Timestamp",
    data_path: str = "data/microstructure",
) -> pd.DataFrame:
    """
    Build GEX surface by querying Parquet greeks snapshots via DuckDB.

    Parameters
    ----------
    conn        : Active DuckDB connection
    symbol      : Underlying ticker
    timestamp   : Exact snapshot timestamp to query
    data_path   : Root of the microstructure data lake

    Returns
    -------
    GEX surface DataFrame (see :func:`build_gex_surface_from_df`).
    """
    parquet_glob = f"{data_path}/options/{symbol}/greeks_snapshots/*.parquet"
    query = f"""
        SELECT
            strike,
            expiration,
            option_type,
            gamma,
            open_interest,
            underlying_price
        FROM '{parquet_glob}'
        WHERE underlying_symbol = '{symbol}'
          AND timestamp = TIMESTAMP '{timestamp}'
          AND gamma IS NOT NULL
          AND open_interest IS NOT NULL
          AND open_interest > 0
    """
    try:
        snapshot = conn.execute(query).fetchdf()
    except Exception as exc:
        raise RuntimeError(f"DuckDB query failed: {exc}") from exc

    if snapshot.empty:
        return pd.DataFrame()

    return build_gex_surface_from_df(snapshot, underlying_symbol=symbol, timestamp=timestamp)


# ---------------------------------------------------------------------------
# GEX analytics helpers
# ---------------------------------------------------------------------------

def gex_flip_level(gex_df: pd.DataFrame) -> float | None:
    """
    Find the strike where net_gex changes sign (the GEX flip point).

    Positive-to-negative flip means dealers go from long to short gamma
    as spot moves up through this level.

    Returns
    -------
    float : Strike of the flip point, or None if no flip exists.
    """
    if gex_df.empty or "net_gex" not in gex_df.columns:
        return None

    df = gex_df.sort_values("strike").reset_index(drop=True)
    # Aggregate across expirations at each strike
    by_strike = df.groupby("strike")["net_gex"].sum().reset_index()
    gex_vals = by_strike["net_gex"].values
    strikes = by_strike["strike"].values

    for i in range(len(gex_vals) - 1):
        if gex_vals[i] * gex_vals[i + 1] < 0:
            # Linear interpolation
            k1, k2 = strikes[i], strikes[i + 1]
            g1, g2 = gex_vals[i], gex_vals[i + 1]
            return float(k1 + (-g1) * (k2 - k1) / (g2 - g1))
    return None


def aggregate_gex_profile(gex_df: pd.DataFrame) -> dict[str, float]:
    """
    Return headline GEX metrics for a snapshot.

    Returns
    -------
    dict with keys: total_call_gex, total_put_gex, total_net_gex,
                    gex_flip_strike, dominant_regime
    """
    if gex_df.empty:
        return {
            "total_call_gex": float("nan"),
            "total_put_gex": float("nan"),
            "total_net_gex": float("nan"),
            "gex_flip_strike": float("nan"),
            "dominant_regime": "unknown",
        }

    total_call = float(gex_df["call_gex"].sum())
    total_put = float(gex_df["put_gex"].sum())
    total_net = float(gex_df["net_gex"].sum())
    flip = gex_flip_level(gex_df)
    regime = "long_gamma" if total_net > 0 else "short_gamma"

    return {
        "total_call_gex": total_call,
        "total_put_gex": total_put,
        "total_net_gex": total_net,
        "gex_flip_strike": flip if flip is not None else float("nan"),
        "dominant_regime": regime,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_gex_surface(gex_df: pd.DataFrame, symbol: str, data_path: str = "data/microstructure") -> None:
    """Persist GEX surface to date-partitioned Parquet."""
    if gex_df.empty:
        return

    df = gex_df.copy()
    df["_date"] = pd.to_datetime(df["timestamp"]).dt.date

    for date, group in df.groupby("_date"):
        path = Path(data_path) / f"options/{symbol}/derived/gex_surface"
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{symbol}_{date}.parquet"
        group.drop(columns="_date").to_parquet(file_path, index=False)
