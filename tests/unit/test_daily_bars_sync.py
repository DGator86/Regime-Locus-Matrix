"""Tests for daily bar sync and freshest-loader selection."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from rlm.data.daily_bars_sync import daily_from_1m_lake, load_best_daily_bars, refresh_symbol_daily_bars
from rlm.data.stock_bars_provider import merge_bars_into_lake


def _minute_bars(start: date, n_days: int) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        ts = pd.Timestamp(day) + pd.Timedelta(hours=15)
        rows.append(
            {
                "timestamp": ts,
                "open": 100.0 + d,
                "high": 101.0 + d,
                "low": 99.0 + d,
                "close": 100.5 + d,
                "volume": 1_000_000,
                "vwap": 100.25 + d,
            }
        )
    return pd.DataFrame(rows)


class TestDailyFrom1mLake:
    def test_resamples_to_daily(self, tmp_path: Path) -> None:
        root = tmp_path
        merge_bars_into_lake("SPY", _minute_bars(date(2026, 5, 1), 5), root=root)
        daily = daily_from_1m_lake("SPY", data_root=root / "data")
        assert len(daily) == 5
        assert daily["close"].iloc[-1] == pytest.approx(104.5)


class TestLoadBestDailyBars:
    def test_prefers_newer_lake_over_stale_csv(self, tmp_path: Path) -> None:
        root = tmp_path
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        stale = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-04-29")],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [711.0],
                "volume": [1.0],
                "vwap": [711.0],
            }
        )
        stale.to_csv(raw / "bars_SPY.csv", index=False)
        merge_bars_into_lake("SPY", _minute_bars(date(2026, 5, 26), 4), root=root)

        best = load_best_daily_bars("SPY", data_root=root / "data")
        assert pd.Timestamp(best["timestamp"].max()).date() >= date(2026, 5, 28)

    def test_prefers_long_fresh_csv_over_short_lake_tail(self, tmp_path: Path) -> None:
        root = tmp_path
        raw = root / "data" / "raw"
        raw.mkdir(parents=True)
        idx = pd.bdate_range("2025-06-02", periods=200)
        pd.DataFrame(
            {
                "timestamp": idx,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
                "vwap": 100.0,
            }
        ).to_csv(raw / "bars_SPY.csv", index=False)
        # Lake ends on the same session but only has a few days of 1m history.
        merge_bars_into_lake("SPY", _minute_bars(idx[-1].date() - timedelta(days=3), 4), root=root)

        best = load_best_daily_bars("SPY", data_root=root / "data")
        assert len(best) >= 180


class TestRefreshSymbolDailyBars:
    def test_yfinance_updates_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sym = "SPY"
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True)

        def fake_fetch(self, start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2026-05-29")],
                    "open": [750.0],
                    "high": [758.0],
                    "low": [749.0],
                    "close": [756.0],
                    "volume": [1e7],
                    "vwap": [755.0],
                }
            )

        monkeypatch.setattr(
            "rlm.data.rolling_store.RollingBarsStore._fetch",
            fake_fetch,
        )
        result = refresh_symbol_daily_bars(sym, tmp_path / "data", today=date(2026, 5, 29))
        assert result.total_rows >= 1
        assert result.last_timestamp is not None
        assert result.last_timestamp.date() == date(2026, 5, 29)
