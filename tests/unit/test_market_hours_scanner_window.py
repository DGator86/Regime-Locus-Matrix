"""Scanner window (09:00–16:00 America/New_York, Mon–Fri) for universe rescans."""

from __future__ import annotations

from datetime import datetime

import pytest

from rlm.utils.market_hours import is_scanner_window_open, scanner_window_label

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc,assignment]


pytestmark = pytest.mark.skipif(ZoneInfo is None, reason="zoneinfo required")


def _et(y: int, m: int, d: int, hour: int, minute: int = 0) -> datetime:
    assert ZoneInfo is not None
    return datetime(y, m, d, hour, minute, tzinfo=ZoneInfo("America/New_York"))


def test_scanner_closed_weekend() -> None:
    sat = _et(2026, 4, 18, 10, 0)  # Saturday
    assert not is_scanner_window_open(_override=sat)
    assert "weekend" in scanner_window_label(_override=sat)


def test_scanner_before_9am_weekday() -> None:
    mon = _et(2026, 4, 20, 8, 59)  # Monday
    assert not is_scanner_window_open(_override=mon)
    assert "before_scanner_open" in scanner_window_label(_override=mon)


def test_scanner_opens_9am_weekday() -> None:
    mon = _et(2026, 4, 20, 9, 0)
    assert is_scanner_window_open(_override=mon)
    assert scanner_window_label(_override=mon) == "scanner_open"


def test_scanner_stops_at_4pm_exclusive() -> None:
    mon = _et(2026, 4, 20, 15, 59)
    assert is_scanner_window_open(_override=mon)
    mon_close = _et(2026, 4, 20, 16, 0)
    assert not is_scanner_window_open(_override=mon_close)
    assert "scanner_close" in scanner_window_label(_override=mon_close)
