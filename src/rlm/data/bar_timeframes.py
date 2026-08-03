"""IBKR bar-size helpers for intraday vs daily history windows."""

from __future__ import annotations

import os
import re


_DAILY_BAR_SIZES = frozenset({"1 day", "1day", "1 d", "1d"})


def is_intraday_bar_size(bar_size: str) -> bool:
    """True when ``bar_size`` is finer than daily (minutes, hours, seconds)."""
    s = bar_size.strip().lower()
    if not s:
        return False
    if s in _DAILY_BAR_SIZES:
        return False
    if re.fullmatch(r"\d+\s*(w|wk|week|month|year|y)s?", s):
        return False
    if "min" in s or "hour" in s or "sec" in s:
        return True
    if re.search(r"\d+\s*h\b", s):
        return True
    return False


def duration_to_calendar_days(duration: str, *, default: int = 30) -> int:
    """Parse IBKR-style duration strings (``30 D``, ``4 W``, ``6 M``) to calendar days."""
    parts = duration.strip().split()
    if len(parts) < 2:
        return default
    try:
        n = int(float(parts[0]))
    except ValueError:
        return default
    unit = parts[1].upper()
    if unit.startswith("D"):
        return max(1, n)
    if unit.startswith("W"):
        return max(1, n * 7)
    if unit.startswith("M"):
        return max(1, n * 30)
    if unit.startswith("Y"):
        return max(1, n * 365)
    return default


def bar_size_step_minutes(bar_size: str) -> float:
    """Approximate bar width in minutes for walk-back cursor steps."""
    s = bar_size.strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(sec|secs|second|seconds|s)\b", s)
    if m:
        return max(1.0 / 60.0, float(m.group(1)) / 60.0)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes|m)\b", s)
    if m:
        return max(1.0, float(m.group(1)))
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs|h)\b", s)
    if m:
        return max(1.0, float(m.group(1)) * 60.0)
    return 5.0


def ibkr_chunk_days_for_bar_size(bar_size: str) -> int:
    """Conservative per-request ``duration`` chunk for IBKR intraday history."""
    step = bar_size_step_minutes(bar_size)
    if step <= 1.0:
        return 6
    if step <= 5.0:
        return 20
    if step <= 15.0:
        return 30
    if step <= 60.0:
        return 60
    return 90


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def apply_intraday_primary_defaults(bar_size: str, duration: str) -> tuple[str, str]:
    """Replace legacy daily primary bars unless ``RLM_ALLOW_DAILY_PRIMARY=1``."""
    if _env_truthy("RLM_ALLOW_DAILY_PRIMARY"):
        return bar_size, duration
    if bar_size.strip().lower() not in _DAILY_BAR_SIZES:
        return bar_size, duration
    upgraded_bar = (os.environ.get("RLM_PRIMARY_BAR_SIZE") or "1 min").strip() or "1 min"
    upgraded_dur = (os.environ.get("RLM_PRIMARY_DURATION") or "30 D").strip() or "30 D"
    return upgraded_bar, upgraded_dur


def clamp_intraday_duration(duration: str, *, min_days: int = 30) -> str:
    """IBKR intraday history is chunked; ensure at least ``min_days`` for MTF/regime fit."""
    if not duration.strip().upper().endswith(" D"):
        return duration
    days = duration_to_calendar_days(duration)
    if days < min_days:
        return f"{min_days} D"
    return duration


# Live seed uses vol_window≈141 (min_periods≈47). ~21 daily bars from ``30 D`` leaves
# b_sigma all-NaN and floors sigma → zero-width debit spreads. Require enough calendar
# days for a full vol window of trading sessions (141 * 7/5 ≈ 198, plus buffer).
DEFAULT_MIN_DAILY_PRIMARY_DAYS = 220


def clamp_daily_duration(
    duration: str,
    *,
    min_days: int = DEFAULT_MIN_DAILY_PRIMARY_DAYS,
) -> str:
    """Ensure daily-primary lookbacks cover forecast move/vol baseline windows."""
    days = duration_to_calendar_days(duration)
    if days < int(min_days):
        return f"{int(min_days)} D"
    # Preserve non-day units when already long enough (e.g. ``1 Y``).
    if duration.strip().upper().endswith(" D"):
        return f"{days} D"
    return duration
