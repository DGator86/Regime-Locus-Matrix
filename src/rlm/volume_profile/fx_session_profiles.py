"""FX-specific session profile utilities."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd
import pytz

from rlm.volume_profile.profile_calculator import calculate_volume_profile

_FX_WINDOWS: dict[str, tuple[time, time]] = {
    "Sydney": (time(17, 0), time(2, 0)),
    "Tokyo": (time(19, 0), time(4, 0)),
    "London": (time(3, 0), time(12, 0)),
    "New_York": (time(8, 0), time(17, 0)),
}


def _bounds(session_name: str, date: datetime) -> tuple[datetime, datetime]:
    if session_name not in _FX_WINDOWS:
        raise ValueError(f"Unsupported FX session '{session_name}'.")
    tz = pytz.timezone("America/New_York")
    start_t, end_t = _FX_WINDOWS[session_name]
    local_date = date.astimezone(tz).date() if pd.Timestamp(date).tzinfo else date.date()
    start_dt = tz.localize(datetime.combine(local_date, start_t))
    end_dt = tz.localize(datetime.combine(local_date, end_t))
    if end_t <= start_t:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def get_fx_session_profile(df: pd.DataFrame, session_name: str, date: datetime) -> dict[str, Any]:
    """Return a volume profile for a named FX session window."""
    if "timestamp" not in df.columns:
        raise ValueError("Expected 'timestamp' column.")

    start_dt, end_dt = _bounds(session_name, date)
    tz = pytz.timezone("America/New_York")

    work = df.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.loc[ts.notna()].copy()
    work["timestamp"] = ts.loc[ts.notna()].dt.tz_convert(tz)

    window = work.loc[(work["timestamp"] >= start_dt) & (work["timestamp"] < end_dt)]
    if window.empty:
        return {
            "poc": float("nan"),
            "value_area_high": float("nan"),
            "value_area_low": float("nan"),
            "value_area_40_high": float("nan"),
            "value_area_40_low": float("nan"),
            "volume_profile_series": pd.Series(dtype=float),
        }
    return calculate_volume_profile(window[["timestamp", "price", "volume"]])


def session_overlap_zones(session1: str, session2: str, date: datetime) -> dict[str, Any]:
    """Return overlap time window between two FX sessions."""
    s1, e1 = _bounds(session1, date)
    s2, e2 = _bounds(session2, date)

    start = max(s1, s2)
    end = min(e1, e2)
    return {
        "session1": session1,
        "session2": session2,
        "overlap_exists": start < end,
        "overlap_start": start if start < end else None,
        "overlap_end": end if start < end else None,
    }


def dominant_session_poc(symbol: str, date: datetime) -> str:
    """Return the session whose POC is closest to current spot price."""
    from rlm.data.microstructure.database.query import MicrostructureDB

    db = MicrostructureDB()
    bars = db.load_underlying_bars(
        symbol.upper(), date.date().isoformat(), date.date().isoformat(), bar_resolution="5s"
    )
    if bars.empty:
        return "Unknown"

    work = bars.rename(columns={"close": "price"}).copy()
    if "price" not in work.columns:
        work["price"] = pd.to_numeric(work.get("close"), errors="coerce")
    spot = float(pd.to_numeric(work["price"], errors="coerce").dropna().iloc[-1])

    session_distances: dict[str, float] = {}
    for session in _FX_WINDOWS:
        profile = get_fx_session_profile(work[["timestamp", "price", "volume"]], session, date)
        poc = profile.get("poc", float("nan"))
        session_distances[session] = abs(spot - float(poc)) if pd.notna(poc) else float("inf")

    return min(session_distances, key=session_distances.get)
