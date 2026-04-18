from __future__ import annotations

import numpy as np
import pandas as pd

_SCORE_COLS = ("S_D", "S_V", "S_L", "S_G")
_COORD_COLS = ("M_D", "M_V", "M_L", "M_G")


def _score_to_coord(s: pd.Series) -> pd.Series:
    """Vectorised map: s ∈ [-1, 1] → coordinate ∈ [0, 10].  NaN propagates."""
    return (5.0 + 5.0 * s).clip(0.0, 10.0)


def _lat_bin(series: pd.Series, low: float, high: float) -> pd.Series:
    """Bin each value into {1, 2, 3}, matching MarketCoordinate.lattice_index logic."""
    return pd.Series(
        np.where(series <= low, 1, np.where(series >= high, 3, 2)),
        index=series.index,
        dtype="Int64",
    )


def add_market_coordinate_columns(
    df: pd.DataFrame,
    low_bound: float = 3.0,
    high_bound: float = 7.0,
) -> pd.DataFrame:
    """
    Add 4D market-coordinate columns to a DataFrame that contains
    composite score columns S_D, S_V, S_L, S_G ∈ [-1, 1].

    New columns produced
    --------------------
    M_D, M_V, M_L, M_G
        Raw coordinates on [0, 10].  NaN propagates from score NaN (warmup rows).

    M_trend_strength
        T = |M_D − 5|, trend intensity independent of direction.

    M_dealer_control
        C = |M_G − 5|, dealer structural influence independent of sign.

    M_alignment
        A_DG = (M_D − 5)·(M_G − 5) ∈ [−25, 25].
        Positive → direction and dealer flow reinforce each other.
        Negative → they oppose each other.

    M_delta_neutral
        δ = √( (M_D−5)² + (M_V−5)² + (M_L−5)² + (M_G−5)² )
        Euclidean distance from the neutral point (5,5,5,5).
        Range [0, 10].  Small = ordinary; large = extreme regime.

    M_lat_D, M_lat_V, M_lat_L, M_lat_G
        Lattice bin index ∈ {1, 2, 3} for each axis (nullable int; NaN on
        warmup rows where coordinates are NaN).
        Bin 1: x ≤ low_bound (default 3)
        Bin 2: low_bound < x < high_bound
        Bin 3: x ≥ high_bound (default 7)

    M_R_trans
        R_trans = ||M_t − M_{t−1}||₂
        Transition magnitude between consecutive bars.
        NaN for the first bar (and after any NaN coordinate row).
        Small → stable regime; large → transition / instability.
    """
    missing = [c for c in _SCORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_market_coordinate_columns: missing score columns {missing}"
        )

    out = df.copy()

    # ---- Coordinate axes --------------------------------------------------
    out["M_D"] = _score_to_coord(out["S_D"])
    out["M_V"] = _score_to_coord(out["S_V"])
    out["M_L"] = _score_to_coord(out["S_L"])
    out["M_G"] = _score_to_coord(out["S_G"])

    # ---- Derived invariants -----------------------------------------------
    d_centered = out["M_D"] - 5.0
    g_centered = out["M_G"] - 5.0

    out["M_trend_strength"] = d_centered.abs()
    out["M_dealer_control"] = g_centered.abs()
    out["M_alignment"] = d_centered * g_centered
    out["M_delta_neutral"] = np.sqrt(
        d_centered**2
        + (out["M_V"] - 5.0) ** 2
        + (out["M_L"] - 5.0) ** 2
        + g_centered**2
    )

    # ---- Lattice index ----------------------------------------------------
    out["M_lat_D"] = _lat_bin(out["M_D"], low_bound, high_bound)
    out["M_lat_V"] = _lat_bin(out["M_V"], low_bound, high_bound)
    out["M_lat_L"] = _lat_bin(out["M_L"], low_bound, high_bound)
    out["M_lat_G"] = _lat_bin(out["M_G"], low_bound, high_bound)

    # ---- Transition magnitude ---------------------------------------------
    prev = out[list(_COORD_COLS)].shift(1)
    out["M_R_trans"] = np.sqrt(
        (out["M_D"] - prev["M_D"]) ** 2
        + (out["M_V"] - prev["M_V"]) ** 2
        + (out["M_L"] - prev["M_L"]) ** 2
        + (out["M_G"] - prev["M_G"]) ** 2
    )

    return out
