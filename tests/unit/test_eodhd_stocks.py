"""EODHD intraday helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from rlm.data.eodhd_stocks import (
    intraday_payload_to_bars,
    is_eodhd_poll_window_open,
    symbol_to_eodhd,
)


def test_symbol_to_eodhd() -> None:
    assert symbol_to_eodhd("spy") == "SPY.US"
    assert symbol_to_eodhd("SPY.US") == "SPY.US"


def test_intraday_payload_to_bars() -> None:
    payload = [
        {
            "datetime": "1700000000",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1000,
        }
    ]
    df = intraday_payload_to_bars(payload)
    assert len(df) == 1
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    assert float(df["close"].iloc[0]) == 100.5


def test_poll_window_weekday_midday() -> None:
    et = ZoneInfo("America/New_York")
    noon = datetime(2026, 5, 19, 12, 0, tzinfo=et)
    assert is_eodhd_poll_window_open(now=noon) is True
    early = datetime(2026, 5, 19, 8, 0, tzinfo=et)
    assert is_eodhd_poll_window_open(now=early) is False
    sat = datetime(2026, 5, 16, 12, 0, tzinfo=et)
    assert is_eodhd_poll_window_open(now=sat) is False


def test_poll_window_ends_at_1630() -> None:
    et = ZoneInfo("America/New_York")
    end = datetime(2026, 5, 19, 16, 29, tzinfo=et)
    after = datetime(2026, 5, 19, 16, 30, tzinfo=et)
    assert is_eodhd_poll_window_open(now=end) is True
    assert is_eodhd_poll_window_open(now=after) is False
