"""Tests for option chain snapshot merge helpers."""

from __future__ import annotations

import pandas as pd

from rlm.datasets.option_history import merge_option_chain_history


def test_merge_appends_and_dedupes() -> None:
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "underlying": ["SPY", "SPY"],
            "expiry": pd.to_datetime(["2024-02-16", "2024-02-16"]),
            "option_type": ["call", "put"],
            "strike": [400.0, 400.0],
            "bid": [1.0, 0.9],
            "ask": [1.1, 1.0],
        }
    )
    new_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "underlying": ["SPY", "SPY"],
            "expiry": pd.to_datetime(["2024-02-16", "2024-02-16"]),
            "option_type": ["call", "put"],
            "strike": [400.0, 400.0],
            "bid": [1.2, 1.0],
            "ask": [1.3, 1.1],
        }
    )
    out = merge_option_chain_history(existing, new_rows)
    assert len(out) == 4
    assert out["timestamp"].min() == pd.Timestamp("2024-01-01")


def test_replace_same_day() -> None:
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 10:00", "2024-01-16 00:00"], format="mixed"),
            "underlying": ["SPY", "SPY"],
            "expiry": pd.to_datetime(["2024-02-16", "2024-02-16"]),
            "option_type": ["call", "call"],
            "strike": [400.0, 400.0],
            "bid": [1.0, 2.0],
            "ask": [1.1, 2.1],
        }
    )
    new_rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-15 16:00"]),
            "underlying": ["SPY"],
            "expiry": pd.to_datetime(["2024-02-16"]),
            "option_type": ["call"],
            "strike": [400.0],
            "bid": [5.0],
            "ask": [5.1],
        }
    )
    out = merge_option_chain_history(
        existing, new_rows, replace_calendar_date=pd.Timestamp("2024-01-15")
    )
    assert len(out) == 2
    jan15 = out[out["timestamp"].dt.date == pd.Timestamp("2024-01-15").date()]
    assert float(jan15["bid"].iloc[0]) == 5.0
    jan16 = out[out["timestamp"].dt.date == pd.Timestamp("2024-01-16").date()]
    assert float(jan16["bid"].iloc[0]) == 2.0
