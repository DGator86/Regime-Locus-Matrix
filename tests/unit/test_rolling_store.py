"""Unit tests for RollingBarsStore — incremental bar append."""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from rlm.data.rolling_store import RollingBarsStore, RollingUpdateResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(start: str, periods: int, freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": 500.0,
            "high": 502.0,
            "low": 498.0,
            "close": 501.0,
            "volume": 1_000_000,
        }
    )


def _write_bars(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _mock_fetch(bars: pd.DataFrame):
    """Return a context manager that patches RollingBarsStore._fetch."""
    return patch.object(RollingBarsStore, "_fetch", return_value=bars)


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoad:
    def test_returns_none_when_no_file(self, tmp_path):
        store = RollingBarsStore("SPY", data_root=tmp_path)
        assert store.load() is None

    def test_returns_sorted_dataframe(self, tmp_path):
        bars = _make_bars("2025-01-06", 5)
        # write in reverse order to verify sort
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", bars.iloc[::-1].reset_index(drop=True))
        store = RollingBarsStore("SPY", data_root=tmp_path)
        loaded = store.load()
        assert loaded is not None
        assert list(loaded["timestamp"]) == sorted(loaded["timestamp"].tolist())


# ---------------------------------------------------------------------------
# update() — already-current guard
# ---------------------------------------------------------------------------


class TestAlreadyCurrent:
    def test_no_fetch_when_last_bar_is_today(self, tmp_path):
        today = date(2026, 5, 17)
        bars = _make_bars("2026-05-12", 5)  # last bar = 2026-05-16... adjust
        # force last bar to today
        bars.iloc[-1, bars.columns.get_loc("timestamp")] = pd.Timestamp(today)
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", bars)
        store = RollingBarsStore("SPY", data_root=tmp_path)

        with patch.object(RollingBarsStore, "_fetch", side_effect=AssertionError("should not fetch")):
            result = store.update(today=today)

        assert result.already_current is True
        assert result.new_rows == 0
        assert result.total_rows == 5


# ---------------------------------------------------------------------------
# update() — incremental append
# ---------------------------------------------------------------------------


class TestIncrementalAppend:
    def test_appends_only_new_bars(self, tmp_path):
        today = date(2026, 5, 17)
        existing = _make_bars("2026-04-01", 20)  # ends 2026-04-20
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", existing)

        # Provider returns bars covering the overlap window + new dates
        fresh = _make_bars("2026-04-16", 27)  # ends 2026-05-12 ish
        store = RollingBarsStore("SPY", data_root=tmp_path)

        with _mock_fetch(fresh):
            store.update(today=today)

        assert result.already_current is False
        # merged rows = union of existing + fresh (deduped)
        loaded = store.load()
        assert loaded is not None
        assert loaded["timestamp"].is_monotonic_increasing
        assert loaded["timestamp"].duplicated().sum() == 0
        assert result.total_rows == len(loaded)
        assert result.new_rows > 0

    def test_overlap_rows_deduplicated(self, tmp_path):
        today = date(2026, 5, 17)
        existing = _make_bars("2026-01-01", 100)
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", existing)

        # fresh overlaps the last 5 rows exactly (overlap window)
        fresh = existing.tail(5).copy()
        fresh["close"] = 999.0  # different value — keep_last should win
        store = RollingBarsStore("SPY", data_root=tmp_path)

        with _mock_fetch(fresh):
            result = store.update(today=today)

        loaded = store.load()
        assert loaded is not None
        assert loaded["timestamp"].duplicated().sum() == 0
        # Overlap rows should reflect the fresh (keep="last") value
        tail_rows = loaded.tail(5)
        assert (tail_rows["close"] == 999.0).all()

    def test_no_existing_file_triggers_full_fetch(self, tmp_path):
        today = date(2026, 5, 17)
        fresh = _make_bars("2024-05-17", 730)
        store = RollingBarsStore("SPY", data_root=tmp_path, lookback_days=730)

        with _mock_fetch(fresh):
            result = store.update(today=today)

        assert result.new_rows == 730
        assert result.total_rows == 730
        assert (tmp_path / "raw" / "bars_SPY.csv").is_file()

    def test_empty_provider_response_leaves_existing_intact(self, tmp_path):
        today = date(2026, 5, 17)
        existing = _make_bars("2026-01-01", 50)
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", existing)
        store = RollingBarsStore("SPY", data_root=tmp_path)

        with _mock_fetch(pd.DataFrame()):
            result = store.update(today=today)

        assert result.new_rows == 0
        assert result.total_rows == 50
        loaded = store.load()
        assert loaded is not None
        assert len(loaded) == 50

    def test_intraday_update_preserves_multiple_bars_on_same_day(self, tmp_path):
        today = date(2026, 5, 17)
        existing = _make_bars("2026-05-17 09:30", 2, freq="5min")
        _write_bars(tmp_path / "raw" / "bars_SPY.csv", existing)
        fresh = _make_bars("2026-05-17 09:35", 2, freq="5min")
        store = RollingBarsStore("SPY", data_root=tmp_path, interval="5m")

        with _mock_fetch(fresh) as fetch:
            result = store.update(today=today)

        loaded = store.load()
        assert loaded is not None
        assert fetch.call_count == 1
        assert result.already_current is False
        assert loaded["timestamp"].dt.strftime("%H:%M").tolist() == ["09:30", "09:35", "09:40"]


# ---------------------------------------------------------------------------
# update() — result fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def test_result_is_rolling_update_result(self, tmp_path):
        today = date(2026, 5, 17)
        fresh = _make_bars("2026-05-01", 10)
        store = RollingBarsStore("SPY", data_root=tmp_path)
        with _mock_fetch(fresh):
            result = store.update(today=today)
        assert isinstance(result, RollingUpdateResult)

    def test_bars_path_points_to_correct_file(self, tmp_path):
        store = RollingBarsStore("SPY", data_root=tmp_path)
        with _mock_fetch(pd.DataFrame()):
            result = store.update(today=date(2026, 5, 17))
        assert result.bars_path == tmp_path / "raw" / "bars_SPY.csv"

    def test_symbol_preserved_in_result(self, tmp_path):
        store = RollingBarsStore("QQQ", data_root=tmp_path)
        with _mock_fetch(pd.DataFrame()):
            result = store.update(today=date(2026, 5, 17))
        assert result.symbol == "QQQ"

    def test_last_timestamp_set_after_update(self, tmp_path):
        today = date(2026, 5, 17)
        fresh = _make_bars("2026-05-01", 10)
        store = RollingBarsStore("SPY", data_root=tmp_path)
        with _mock_fetch(fresh):
            result = store.update(today=today)
        assert result.last_timestamp is not None
        assert result.last_timestamp >= pd.Timestamp("2026-05-01")


# ---------------------------------------------------------------------------
# merge logic
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_with_none_existing_returns_fresh(self):
        fresh = _make_bars("2026-01-01", 10)
        merged = RollingBarsStore._merge(None, fresh)
        assert len(merged) == 10

    def test_merge_sorts_by_timestamp(self):
        a = _make_bars("2026-01-01", 5)
        b = _make_bars("2025-12-25", 5)  # older — should appear first
        merged = RollingBarsStore._merge(a, b)
        assert merged["timestamp"].is_monotonic_increasing

    def test_merge_deduplicates_on_timestamp(self):
        a = _make_bars("2026-01-01", 10)
        b = _make_bars("2026-01-08", 5)  # overlaps last 3 of a
        merged = RollingBarsStore._merge(a, b)
        assert merged["timestamp"].duplicated().sum() == 0

    def test_merge_preserves_intraday_timestamps_when_requested(self):
        fresh = _make_bars("2026-05-17 09:30", 3, freq="5min")
        merged = RollingBarsStore._merge(None, fresh, normalize_dates=False)

        assert len(merged) == 3
        assert merged["timestamp"].dt.strftime("%H:%M").tolist() == ["09:30", "09:35", "09:40"]


class TestFetch:
    def test_fetch_preserves_intraday_timestamps(self, monkeypatch, tmp_path):
        idx = pd.date_range("2026-05-17 09:30", periods=2, freq="5min")
        raw = pd.DataFrame(
            {
                "Open": [500.0, 501.0],
                "High": [502.0, 503.0],
                "Low": [499.0, 500.0],
                "Close": [501.0, 502.0],
                "Volume": [1_000_000, 1_100_000],
            },
            index=idx,
        )

        class FakeTicker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, **kwargs):
                return raw

        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
        store = RollingBarsStore("SPY", data_root=tmp_path, interval="5m")

        fetched = store._fetch(date(2026, 5, 17), date(2026, 5, 18))

        assert fetched["timestamp"].dt.strftime("%H:%M").tolist() == ["09:30", "09:35"]
