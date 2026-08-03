"""Unit tests for IBKR bar-size / intraday window helpers."""

from __future__ import annotations

import pytest

from rlm.data.bar_timeframes import (
    apply_intraday_primary_defaults,
    bar_size_step_minutes,
    clamp_daily_duration,
    clamp_intraday_duration,
    duration_to_calendar_days,
    ibkr_chunk_days_for_bar_size,
    is_intraday_bar_size,
)


@pytest.mark.parametrize(
    ("bar_size", "expected"),
    [
        ("1 min", True),
        ("5 mins", True),
        ("1 hour", True),
        ("1 day", False),
        ("1 week", False),
    ],
)
def test_is_intraday_bar_size(bar_size: str, expected: bool) -> None:
    assert is_intraday_bar_size(bar_size) is expected


def test_duration_to_calendar_days() -> None:
    assert duration_to_calendar_days("30 D") == 30
    assert duration_to_calendar_days("2 W") == 14


def test_clamp_intraday_duration() -> None:
    assert clamp_intraday_duration("10 D") == "30 D"
    assert clamp_intraday_duration("45 D") == "45 D"


def test_clamp_daily_duration() -> None:
    assert clamp_daily_duration("30 D") == "220 D"
    assert clamp_daily_duration("400 D") == "400 D"


def test_ibkr_chunk_days_for_1min() -> None:
    assert ibkr_chunk_days_for_bar_size("1 min") <= 7
    assert ibkr_chunk_days_for_bar_size("5 mins") == 20


def test_bar_size_step_minutes() -> None:
    assert bar_size_step_minutes("1 min") == 1.0
    assert bar_size_step_minutes("5 mins") == 5.0


def test_apply_intraday_primary_defaults_upgrades_daily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_ALLOW_DAILY_PRIMARY", raising=False)
    bar, dur = apply_intraday_primary_defaults("1 day", "180 D")
    assert bar == "1 min"
    assert dur == "30 D"


def test_apply_intraday_primary_defaults_respects_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_ALLOW_DAILY_PRIMARY", "1")
    bar, dur = apply_intraday_primary_defaults("1 day", "180 D")
    assert bar == "1 day"
    assert dur == "180 D"


def test_apply_intraday_primary_defaults_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RLM_ALLOW_DAILY_PRIMARY", raising=False)
    monkeypatch.setenv("RLM_PRIMARY_BAR_SIZE", "5 mins")
    monkeypatch.setenv("RLM_PRIMARY_DURATION", "20 D")
    bar, dur = apply_intraday_primary_defaults("1day", "90 D")
    assert bar == "5 mins"
    assert dur == "20 D"
    monkeypatch.delenv("RLM_PRIMARY_BAR_SIZE", raising=False)
    monkeypatch.delenv("RLM_PRIMARY_DURATION", raising=False)
