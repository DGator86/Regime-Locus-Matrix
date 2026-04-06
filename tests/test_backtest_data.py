"""Tests for rolling backtest dataset helpers."""

from __future__ import annotations

import pandas as pd

from rlm.backtest.walkforward import WalkForwardConfig
from rlm.datasets.backtest_data import (
    bars_to_csv_frame,
    rolling_window_manifest,
    synthetic_5m_bars_range,
    synthetic_bars_demo,
    synthetic_option_chain_from_bars,
    synthetic_option_chain_intraday_from_bars,
    us_equity_rth_5m_index,
)


def test_synthetic_bars_and_chain_align() -> None:
    bars = synthetic_bars_demo("2026-01-10", periods=50)
    chain = synthetic_option_chain_from_bars(bars, underlying="SPY")
    assert len(bars) == 50
    assert chain["timestamp"].nunique() == 50
    assert set(chain["underlying"].unique()) == {"SPY"}
    assert {"bid", "ask", "strike", "expiry"}.issubset(chain.columns)


def test_rolling_window_manifest_counts() -> None:
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    cfg = WalkForwardConfig(is_window=100, oos_window=50, step_size=50)
    m = rolling_window_manifest(idx, cfg)
    assert len(m) == 2
    assert m.iloc[0]["window_id"] == 0
    assert m.iloc[0]["oos_bar_end_idx"] == 150
    assert m.iloc[1]["is_bar_start_idx"] == 50


def test_bars_to_csv_roundtrip_index() -> None:
    bars = synthetic_bars_demo("2026-02-01", periods=10)
    csv_df = bars_to_csv_frame(bars)
    assert "timestamp" in csv_df.columns
    back = csv_df.set_index("timestamp")
    assert len(back) == len(bars)


def test_us_equity_rth_5m_index_one_week() -> None:
    idx = us_equity_rth_5m_index(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09"))
    assert len(idx) == 5 * 78


def test_synthetic_5m_and_intraday_chain() -> None:
    bars = synthetic_5m_bars_range(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"))
    ch = synthetic_option_chain_intraday_from_bars(bars, underlying="SPY")
    assert len(bars) == 3 * 78
    assert ch["timestamp"].nunique() == len(bars)
    assert len(ch) == len(bars) * 12
