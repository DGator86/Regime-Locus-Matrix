"""US equity Regular Trading Hours (RTH) utilities.

All functions operate in US/Eastern time. No external dependencies beyond the
standard library and ``datetime``; ``zoneinfo`` is used when available (Python
3.9+), falling back to ``dateutil`` if installed, otherwise UTC offset -5/-4.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone, tzinfo
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

# RTH session boundaries (Eastern)
_RTH_OPEN = time(9, 30, 0)
_RTH_CLOSE = time(16, 0, 0)

# Universe / rescan window (US Eastern legal time via ``_eastern_tz`` — EST in winter, EDT in summer).
# Mon–Fri inclusive; ``start`` inclusive, ``end`` exclusive (scanner stops at 16:00).
_SCANNER_START = time(9, 0, 0)
_SCANNER_END = time(16, 0, 0)


def _now_eastern() -> datetime:
    return datetime.now(_EASTERN)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_rth_now(*, _override: Optional[datetime] = None) -> bool:
    """Return True if the current moment is within US equity RTH (Mon–Fri, 09:30–16:00 ET).

    Pass ``_override`` in tests to inject a fixed eastern datetime.
    """
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now.time().replace(second=0, microsecond=0)
    return _RTH_OPEN <= t < _RTH_CLOSE


def minutes_into_session(*, _override: Optional[datetime] = None) -> int:
    """Minutes elapsed since 09:30 ET today.  Negative before open; ≥390 after close."""
    now = _override if _override is not None else _now_eastern()
    open_dt = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return int((now - open_dt).total_seconds() / 60)


def minutes_to_session_end(*, _override: Optional[datetime] = None) -> int:
    """Minutes remaining until 16:00 ET.  Negative after close."""
    now = _override if _override is not None else _now_eastern()
    close_dt = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return int((close_dt - now).total_seconds() / 60)


def entry_window_open(
    *,
    buffer_open_minutes: int = 15,
    buffer_close_minutes: int = 30,
    _override: Optional[datetime] = None,
) -> bool:
    """Return True when it is safe to enter new positions.

    Blocks entries:
    - Outside RTH (weekends, pre-market, after-hours)
    - Within the first ``buffer_open_minutes`` of the session (09:30–09:44 default)
    - Within the last ``buffer_close_minutes`` of the session (15:30–15:59 default)

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
    """True Mon–Fri when local Eastern clock is in ``[09:00, 16:00)``.

    Used to gate periodic universe rescans (``run_everything`` master loop).
    This is **not** identical to RTH (09:30–16:00); it matches a 9:00–16:00 ET scan day.
    """
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:
        return False
    t = now.time().replace(second=0, microsecond=0)
    return _SCANNER_START <= t < _SCANNER_END


def scanner_window_label(*, _override: Optional[datetime] = None) -> str:
    """Human-readable state for :func:`is_scanner_window_open`."""
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:
        return "weekend (scanner off)"
    t = now.time().replace(second=0, microsecond=0)
    if t < _SCANNER_START:
        return f"before_scanner_open ({_SCANNER_START.strftime('%H:%M')} ET)"
    if t >= _SCANNER_END:
        return f"at_or_after_scanner_close ({_SCANNER_END.strftime('%H:%M')} ET)"
    return "scanner_open"


def session_label(*, _override: Optional[datetime] = None) -> str:
    """Human-readable session state for logging."""
    now = _override if _override is not None else _now_eastern()
    if now.weekday() >= 5:
        return "weekend"
    into = minutes_into_session(_override=_override)
    to_end = minutes_to_session_end(_override=_override)
    if into < 0:
        return f"pre_market ({-into}m before open)"
    if to_end <= 0:
        return f"after_hours ({-to_end}m after close)"
    return f"rth ({into}m in, {to_end}m remaining)"
