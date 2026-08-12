"""Host watchdog must not undo rlm-market-hours-stop for trading units."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from rlm.utils.host_watchdog_policy import should_auto_restart_watched_service
from rlm.utils.market_hours import is_market_service_window_open

ET = ZoneInfo("America/New_York")


def test_market_service_window_matches_open_close_timers() -> None:
    assert is_market_service_window_open(_override=datetime(2026, 8, 12, 8, 59, tzinfo=ET)) is False
    assert is_market_service_window_open(_override=datetime(2026, 8, 12, 9, 0, tzinfo=ET)) is True
    assert is_market_service_window_open(_override=datetime(2026, 8, 12, 16, 29, tzinfo=ET)) is True
    assert is_market_service_window_open(_override=datetime(2026, 8, 12, 16, 30, tzinfo=ET)) is False
    assert is_market_service_window_open(_override=datetime(2026, 8, 15, 11, 0, tzinfo=ET)) is False  # Saturday


def test_watchdog_skips_master_trader_after_market_close() -> None:
    after_close = datetime(2026, 8, 12, 16, 45, tzinfo=ET)
    assert should_auto_restart_watched_service("rlm-master-trader", _override=after_close) is False
    assert should_auto_restart_watched_service("rlm-challenge-loop", _override=after_close) is False
    # Always-on units still eligible.
    assert should_auto_restart_watched_service("ollama", _override=after_close) is True
    assert should_auto_restart_watched_service("regime-locus-crew", _override=after_close) is True


def test_watchdog_allows_master_trader_restart_during_service_window() -> None:
    midday = datetime(2026, 8, 12, 11, 0, tzinfo=ET)
    assert should_auto_restart_watched_service("rlm-master-trader", _override=midday) is True
    assert should_auto_restart_watched_service("rlm-master-trader.service", _override=midday) is True
