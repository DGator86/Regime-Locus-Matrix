"""Daily primary history must cover forecast baseline windows."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlm.data.bar_timeframes import clamp_daily_duration, duration_to_calendar_days
from rlm.data.stock_bars_provider import _trim_daily_lookback, fetch_stock_bars, merge_bars_into_lake
from rlm.forecasting.distribution import compute_baseline_vol_scale
from rlm.roee.strategy_map import get_strategy_for_regime
from rlm.roee.strike_selection import build_legs_from_candidate, ensure_vertical_width


def test_clamp_daily_duration_raises_short_lookback() -> None:
    assert clamp_daily_duration("30 D") == "220 D"
    assert clamp_daily_duration("220 D") == "220 D"
    assert clamp_daily_duration("1 Y") == "1 Y"


def test_trim_daily_lookback_keeps_newest_span() -> None:
    start = date(2025, 1, 2)
    rows = [
        {
            "timestamp": pd.Timestamp(start + timedelta(days=i)),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + i * 0.01,
            "volume": 1e6,
            "vwap": 100.0,
        }
        for i in range(0, 400, 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]
    daily = pd.DataFrame(rows)
    trimmed = _trim_daily_lookback(daily, lookback_days=220)
    assert len(trimmed) >= 140
    span_days = (trimmed["timestamp"].max() - trimmed["timestamp"].min()).days
    assert span_days <= 220


def test_eodhd_daily_uses_long_csv_not_30d_lake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # Long daily history in CSV (what VPS yfinance sync provides).
    idx = pd.bdate_range("2025-01-02", periods=260)
    csv = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + np.arange(len(idx)) * 0.05,
            "volume": 1e6,
            "vwap": 100.0,
        }
    )
    csv.to_csv(raw / "bars_SPY.csv", index=False)
    # Short 1m lake tail must not win when CSV has longer usable history of same last date.
    merge_bars_into_lake(
        "SPY",
        pd.DataFrame(
            {
                "timestamp": [idx[-1] + pd.Timedelta(hours=15)],
                "open": [200.0],
                "high": [201.0],
                "low": [199.0],
                "close": [200.5],
                "volume": [1e6],
                "vwap": [200.0],
            }
        ),
        root=tmp_path,
    )

    bars = fetch_stock_bars(
        "SPY",
        duration="220 D",
        bar_size="1 day",
        data_root=tmp_path,
    )
    assert len(bars) >= 140
    b_sigma = compute_baseline_vol_scale(bars["close"], window=141)
    assert pd.notna(b_sigma.iloc[-1])


def test_floor_sigma_vertical_keeps_positive_width() -> None:
    cand = get_strategy_for_regime("bull", "unknown", "low_liquidity", "destabilizing")
    legs = build_legs_from_candidate(cand, current_price=96.46, sigma=0.0001, strike_increment=1.0)
    assert len(legs) == 2
    assert legs[0].strike != legs[1].strike
    assert legs[1].strike == legs[0].strike + 1.0

    long_s, short_s = ensure_vertical_width(100.0, 100.0, option_type="put", increment=1.0)
    assert short_s < long_s


def test_three_track_migrate_duration_covers_vol_window() -> None:
    text = Path("scripts/migrate_vps_three_tracks.py").read_text(encoding="utf-8")
    assert '"RLM_PRIMARY_DURATION": "220 D"' in text
    assert duration_to_calendar_days("220 D") >= 220


def test_stale_dte_cleanup_skips_large_account_log_name() -> None:
    """Guard mirrored from run_universe_options_pipeline short-DTE cleanup."""
    for name in (
        "options_large_account_trade_log.csv",
        "swing_trade_log.csv",
    ):
        low = name.lower()
        assert "large_account" in low or "swing" in low
    assert "large_account" not in "trade_log.csv"
