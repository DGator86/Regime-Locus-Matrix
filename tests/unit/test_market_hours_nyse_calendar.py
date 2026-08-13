"""NYSE holiday / early-close calendar for RTH trading gates."""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from rlm.utils.market_hours import (
    entry_window_open,
    is_nyse_full_holiday,
    is_rth_now,
    is_scanner_window_open,
    minutes_to_session_end,
    nyse_session_close_time,
    scanner_window_label,
    session_label,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc,assignment]


pytestmark = pytest.mark.skipif(ZoneInfo is None, reason="zoneinfo required")


def _et(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    assert ZoneInfo is not None
    return datetime(y, m, d, hour, minute, tzinfo=ZoneInfo("America/New_York"))


def test_labor_day_2026_is_not_rth() -> None:
    """Weekday clock alone would treat Labor Day 14:00 ET as RTH; NYSE is closed."""
    holiday_afternoon = _et(2026, 9, 7, 14, 0)
    assert is_nyse_full_holiday(date(2026, 9, 7))
    assert not is_rth_now(_override=holiday_afternoon)
    assert not is_scanner_window_open(_override=holiday_afternoon)
    assert not entry_window_open(_override=holiday_afternoon)
    assert session_label(_override=holiday_afternoon) == "nyse_holiday"
    assert "nyse_holiday" in scanner_window_label(_override=holiday_afternoon)


def test_independence_day_observed_2026_is_not_rth() -> None:
    friday = _et(2026, 7, 3, 10, 0)
    assert not is_rth_now(_override=friday)
    assert not entry_window_open(_override=friday)


def test_regular_monday_still_rth() -> None:
    mon = _et(2026, 8, 17, 14, 0)
    assert is_rth_now(_override=mon)
    assert is_scanner_window_open(_override=mon)
    assert entry_window_open(_override=mon)


def test_thanksgiving_friday_early_close_2026() -> None:
    """Fri 2026-11-27 cash close is 13:00 ET; 14:00 must not look like RTH."""
    assert nyse_session_close_time(date(2026, 11, 27)) == time(13, 0, 0)
    morning = _et(2026, 11, 27, 11, 0)
    assert is_rth_now(_override=morning)
    assert is_scanner_window_open(_override=morning)
    assert entry_window_open(_override=morning)

    after_cash_close = _et(2026, 11, 27, 14, 0)
    assert not is_rth_now(_override=after_cash_close)
    assert not is_scanner_window_open(_override=after_cash_close)
    assert not entry_window_open(_override=after_cash_close)
    assert "13:00" in scanner_window_label(_override=after_cash_close)


def test_early_close_entry_buffer_uses_1300_bell() -> None:
    """Default 30m close buffer must fire at 12:30 on a 13:00 early-close day."""
    before_buffer = _et(2026, 11, 27, 12, 29)
    assert entry_window_open(_override=before_buffer)
    in_buffer = _et(2026, 11, 27, 12, 30)
    assert is_rth_now(_override=in_buffer)
    assert not entry_window_open(_override=in_buffer)
    assert minutes_to_session_end(_override=in_buffer) == 30


def test_christmas_eve_2026_early_close() -> None:
    assert not is_nyse_full_holiday(date(2026, 12, 24))
    noon = _et(2026, 12, 24, 12, 0)
    assert is_rth_now(_override=noon)
    after = _et(2026, 12, 24, 13, 0)
    assert not is_rth_now(_override=after)
