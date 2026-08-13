"""US equity Regular Trading Hours (RTH) utilities.

All functions operate in US/Eastern time. No external dependencies beyond the
standard library and ``datetime``; ``zoneinfo`` is used when available (Python
3.9+), falling back to ``dateutil`` if installed, otherwise UTC offset -5/-4.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Optional

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:
    _ZoneInfo = None  # type: ignore[assignment,misc]

try:
    from dateutil.tz import gettz as _dateutil_gettz  # type: ignore[import-untyped]
except ImportError:
    _dateutil_gettz = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _eastern_tz() -> tzinfo:
    """Return a US/Eastern tzinfo object, trying zoneinfo → dateutil → fixed offset."""
    if _ZoneInfo is not None:
        return _ZoneInfo("America/New_York")
    if _dateutil_gettz is not None:
        tz = _dateutil_gettz("America/New_York")
        if tz is not None:
            return tz
    # Last resort: fixed UTC-5 (EST, ignores DST — close enough for a gate)
    return timezone(timedelta(hours=-5))


_EASTERN = _eastern_tz()

# RTH session boundaries (Eastern) — also used for universe rescans (NYSE cash clock window).
_RTH_OPEN = time(9, 30, 0)
_RTH_CLOSE = time(16, 0, 0)
# NYSE cash early close (eligible options 13:15 ET). Use 13:00 so equity and options
# gates stop together; trading after the cash close is never treated as RTH.
_RTH_EARLY_CLOSE = time(13, 0, 0)

# Official NYSE Group full-session holidays (observed dates). Refresh from
# https://www.nyse.com/markets/hours-calendars when a new year is published.
# Years outside this table keep weekday-clock behaviour (fail-open).
_NYSE_FULL_HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),  # Independence Day observed (July 4 is Saturday)
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),  # Juneteenth observed
        date(2027, 7, 5),  # Independence Day observed
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),  # Christmas Day observed
        # 2028 (New Year's Day falls on Saturday — not observed)
        date(2028, 1, 17),
        date(2028, 2, 21),
        date(2028, 4, 14),
        date(2028, 5, 29),
        date(2028, 6, 19),
        date(2028, 7, 4),
        date(2028, 9, 4),
        date(2028, 11, 23),
        date(2028, 12, 25),
    }
)

# 1:00 p.m. ET cash early closes (ICE/NYSE Group holiday calendar).
_NYSE_EARLY_CLOSE_DATES: frozenset[date] = frozenset(
    {
        date(2025, 11, 28),  # day after Thanksgiving
        date(2025, 12, 24),  # Christmas Eve
        date(2026, 11, 27),  # day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
        date(2027, 11, 26),  # day after Thanksgiving
        date(2028, 7, 3),  # day before Independence Day
        date(2028, 11, 24),  # day after Thanksgiving
    }
)


def _now_eastern() -> datetime:
    return datetime.now(_EASTERN)


def is_nyse_full_holiday(day: date) -> bool:
    """True on NYSE full-session closures (weekends are handled separately)."""
    return day in _NYSE_FULL_HOLIDAYS


def nyse_session_close_time(day: date) -> time:
    """Cash RTH close for ``day``: 13:00 ET on listed early-close dates, else 16:00 ET."""
    if day in _NYSE_EARLY_CLOSE_DATES:
        return _RTH_EARLY_CLOSE
    return _RTH_CLOSE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_rth_now(*, _override: Optional[datetime] = None) -> bool:
    """Return True if the current moment is within NYSE cash RTH.

    Regular session: Mon–Fri, 09:30–16:00 ET, excluding NYSE full holidays.
    Early-close dates use 09:30–13:00 ET.

    Pass ``_override`` in tests to inject a fixed eastern datetime.
    """
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    day = now.date()
    if is_nyse_full_holiday(day):
        return False
    t = now.time().replace(second=0, microsecond=0)
    return _RTH_OPEN <= t < nyse_session_close_time(day)


def minutes_into_session(*, _override: Optional[datetime] = None) -> int:
    """Minutes elapsed since 09:30 ET today.  Negative before open; ≥390 after close."""
    now = _override if _override is not None else _now_eastern()
    open_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return int((now - open_dt).total_seconds() / 60)


def minutes_to_session_end(*, _override: Optional[datetime] = None) -> int:
    """Minutes remaining until that day's NYSE cash close.  Negative after close."""
    now = _override if _override is not None else _now_eastern()
    close = nyse_session_close_time(now.date())
    close_dt = now.replace(hour=close.hour, minute=close.minute, second=0, microsecond=0)
    return int((close_dt - now).total_seconds() / 60)


def entry_window_open(
    *,
    buffer_open_minutes: int = 15,
    buffer_close_minutes: int = 30,
    _override: Optional[datetime] = None,
) -> bool:
    """Return True when it is safe to enter new positions.

    Blocks entries:
    - Outside RTH (weekends, NYSE holidays, pre-market, after-hours, early-close tail)
    - Within the first ``buffer_open_minutes`` of the session (09:30–09:44 default)
    - Within the last ``buffer_close_minutes`` of the session (15:30–15:59 default;
      12:30–12:59 on 13:00 ET early-close days)

    These buffers guard against the illiquid open-auction spread spike and against
    entering theta-heavy positions too close to the close.
    """
    if not is_rth_now(_override=_override):
        return False
    into = minutes_into_session(_override=_override)
    if into < buffer_open_minutes:
        return False
    to_end = minutes_to_session_end(_override=_override)
    if to_end <= buffer_close_minutes:
        return False
    return True


def is_friday_afternoon(
    *,
    cutoff_minutes_before_close: int = 60,
    _override: Optional[datetime] = None,
) -> bool:
    """Return True on Friday within ``cutoff_minutes_before_close`` of close.

    Useful for extra caution on short-dated positions that could expire over the
    weekend (calendar spreads, 1DTE entering Thursday, etc.).
    """
    now = _override if _override is not None else _now_eastern()
    if now.weekday() != 4:  # Friday
        return False
    return minutes_to_session_end(_override=_override) <= cutoff_minutes_before_close


def is_scanner_window_open(*, _override: Optional[datetime] = None) -> bool:
    """True during NYSE cash RTH (same half-open interval as :func:`is_rth_now`).

    Used to gate periodic universe rescans (``run_everything`` master loop).
    Closed on weekends, NYSE full holidays, and after the early-close bell.
    """
    return is_rth_now(_override=_override)


def scanner_window_label(*, _override: Optional[datetime] = None) -> str:
    """Human-readable state for :func:`is_scanner_window_open`."""
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:
        return "weekend (scanner off)"
    if is_nyse_full_holiday(now.date()):
        return "nyse_holiday (scanner off)"
    t = now.time().replace(second=0, microsecond=0)
    close = nyse_session_close_time(now.date())
    if t < _RTH_OPEN:
        return f"before_scanner_open ({_RTH_OPEN.strftime('%H:%M')} ET)"
    if t >= close:
        return f"at_or_after_scanner_close ({close.strftime('%H:%M')} ET)"
    return "scanner_open"


def session_label(*, _override: Optional[datetime] = None) -> str:
    """Human-readable session state for logging."""
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:
        return "weekend"
    if is_nyse_full_holiday(now.date()):
        return "nyse_holiday"
    into = minutes_into_session(_override=_override)
    to_end = minutes_to_session_end(_override=_override)
    if into < 0:
        return f"pre_market ({-into}m before open)"
    if to_end <= 0:
        return f"after_hours ({-to_end}m after close)"
    return f"rth ({into}m in, {to_end}m remaining)"
