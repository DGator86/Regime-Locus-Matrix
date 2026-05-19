from __future__ import annotations

import pandas as pd
import pytest

from rlm.data.lake import stock_1m_parquet
from rlm.data.stock_bars_provider import fetch_stock_bars, load_stock_1m_from_lake, merge_bars_into_lake


def _bars(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="min")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": range(periods),
            "high": range(1, periods + 1),
            "low": range(periods),
            "close": range(1, periods + 1),
            "volume": [100.0] * periods,
            "vwap": range(1, periods + 1),
        }
    )


def test_merge_write_failure_leaves_existing_lake_intact(tmp_path, monkeypatch):
    existing = _bars("2026-05-19 09:30", 3)
    merge_bars_into_lake("SPY", existing, root=tmp_path)
    before = load_stock_1m_from_lake("SPY", root=tmp_path, lookback_days=None)

    def fail_to_parquet(*args, **kwargs):
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        merge_bars_into_lake("SPY", _bars("2026-05-19 09:33", 2), root=tmp_path)

    after = load_stock_1m_from_lake("SPY", root=tmp_path, lookback_days=None)
    pd.testing.assert_frame_equal(after, before)
    lake_path = stock_1m_parquet("SPY", duration_slug="full", root=tmp_path)
    assert list(lake_path.parent.glob(f"{lake_path.name}.*.tmp")) == []


def test_eodhd_source_does_not_return_partial_lake_after_backfill_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.setenv("RLM_EODHD_MIN_LAKE_BARS", "10")
    monkeypatch.setenv("EODHD_API_KEY", "token")
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)
    merge_bars_into_lake("SPY", _bars("2026-05-19 09:30", 2), root=tmp_path)

    def fail_backfill(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("rlm.data.stock_bars_provider.fetch_intraday_lookback_days", fail_backfill)
    monkeypatch.setattr(
        "rlm.data.stock_bars_provider._fetch_ibkr_intraday",
        lambda *args, **kwargs: pytest.fail("IBKR fallback should be disabled"),
    )

    out = fetch_stock_bars(
        "SPY",
        duration="30 D",
        bar_size="1 min",
        data_root=tmp_path,
    )

    assert out.empty


def test_eodhd_source_without_key_does_not_fall_through_to_ibkr(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.setenv("RLM_EODHD_MIN_LAKE_BARS", "10")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    monkeypatch.delenv("EOD_HISTORICAL_API_KEY", raising=False)
    monkeypatch.delenv("EODHD_API_TOKEN", raising=False)
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)

    monkeypatch.setattr(
        "rlm.data.stock_bars_provider._fetch_ibkr_intraday",
        lambda *args, **kwargs: pytest.fail("IBKR fallback should be disabled"),
    )

    out = fetch_stock_bars(
        "SPY",
        duration="30 D",
        bar_size="1 min",
        data_root=tmp_path,
    )

    assert out.empty


def test_auto_source_uses_ibkr_instead_of_partial_lake(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "auto")
    monkeypatch.setenv("RLM_EODHD_MIN_LAKE_BARS", "10")
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    merge_bars_into_lake("SPY", _bars("2026-05-19 09:30", 2), root=tmp_path)
    ibkr = _bars("2026-05-19 10:00", 10)

    monkeypatch.setattr("rlm.data.stock_bars_provider._fetch_ibkr_intraday", lambda *args, **kwargs: ibkr)

    out = fetch_stock_bars(
        "SPY",
        duration="30 D",
        bar_size="1 min",
        data_root=tmp_path,
    )

    pd.testing.assert_frame_equal(out, ibkr)
