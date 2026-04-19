"""Session-aware volume-profile helpers."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

import pandas as pd
import pytz

from rlm.volume_profile.profile_calculator import calculate_volume_profile

_SESSION_WINDOWS: dict[str, tuple[time, time]] = {
    "RTH": (time(9, 30), time(16, 0)),
    "ETH": (time(18, 0), time(9, 30)),
    "London": (time(3, 0), time(12, 0)),
    "New_York": (time(8, 0), time(17, 0)),
    "Tokyo": (time(19, 0), time(4, 0)),
    "Sydney": (time(17, 0), time(2, 0)),
}


def _window_bounds(
    session_name: str, date: datetime, tz: pytz.BaseTzInfo
) -> tuple[datetime, datetime]:
    if session_name not in _SESSION_WINDOWS:
        raise ValueError(f"Unsupported session '{session_name}'.")

    start_t, end_t = _SESSION_WINDOWS[session_name]
    local_date = date.astimezone(tz).date()

    start_dt = tz.localize(datetime.combine(local_date, start_t))
    end_dt = tz.localize(datetime.combine(local_date, end_t))

    if end_t <= start_t:
        if session_name == "ETH":
            start_dt = start_dt - timedelta(days=1)
        else:
            end_dt = end_dt + timedelta(days=1)

    return start_dt, end_dt


def get_session_profile(df: pd.DataFrame, session_name: str, date: datetime) -> dict[str, Any]:
    """Compute a session-specific volume profile for a given date."""

    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in input DataFrame.")

    tz = pytz.timezone("America/New_York")
    start_dt, end_dt = _window_bounds(session_name=session_name, date=date, tz=tz)

    work = df.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.loc[ts.notna()].copy()
    work["timestamp"] = ts.loc[ts.notna()].dt.tz_convert(tz)

    mask = (work["timestamp"] >= start_dt) & (work["timestamp"] < end_dt)
    session_df = work.loc[mask, ["timestamp", "price", "volume"]]
    if session_df.empty:
        return {
            "poc": float("nan"),
            "value_area_high": float("nan"),
            "value_area_low": float("nan"),
            "value_area_40_high": float("nan"),
            "value_area_40_low": float("nan"),
            "volume_profile_series": pd.Series(dtype=float),
        }

    return calculate_volume_profile(session_df)


def overlap_zones(
    session1_profile: dict[str, Any], session2_profile: dict[str, Any]
) -> dict[str, float | bool]:
    """Return the overlap region of two session value areas."""

    low = max(
        float(session1_profile.get("value_area_low", float("nan"))),
        float(session2_profile.get("value_area_low", float("nan"))),
    )
    high = min(
        float(session1_profile.get("value_area_high", float("nan"))),
        float(session2_profile.get("value_area_high", float("nan"))),
    )

    overlap_exists = low <= high
    return {
        "overlap_exists": overlap_exists,
        "overlap_low": low if overlap_exists else float("nan"),
        "overlap_high": high if overlap_exists else float("nan"),
    }
