from __future__ import annotations

import pandas as pd
import pytest

from rlm.data import stock_bars_provider as provider


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


def test_merge_uses_atomic_save_after_read_merge(tmp_path, monkeypatch):
    existing = _bars("2026-05-19 09:30", 3)
    fresh = _bars("2026-05-19 09:32", 3)
    saved: dict[str, object] = {}

    monkeypatch.setattr(provider, "load_stock_1m_from_lake", lambda *args, **kwargs: existing)

    def save(df: pd.DataFrame, path, *, index: bool) -> None:
        saved["df"] = df.copy()
        saved["path"] = path
        saved["index"] = index

    monkeypatch.setattr(provider, "save_parquet", save)

    out = provider.merge_bars_into_lake("SPY", fresh, root=tmp_path)

    assert saved["path"] == provider._lake_path("SPY", root=tmp_path)
    assert saved["index"] is False
    assert out["timestamp"].duplicated().sum() == 0
    assert len(out) == 5
    pd.testing.assert_frame_equal(saved["df"], out)


def test_eodhd_source_does_not_return_partial_lake_after_backfill_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.setenv("RLM_EODHD_MIN_LAKE_BARS", "10")
    monkeypatch.setenv("EODHD_API_KEY", "token")
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)
    monkeypatch.setattr(provider, "load_stock_1m_from_lake", lambda *args, **kwargs: _bars("2026-05-19 09:30", 2))

    def fail_backfill(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(provider, "fetch_intraday_lookback_days", fail_backfill)
    monkeypatch.setattr(
        provider,
        "_fetch_ibkr_intraday",
        lambda *args, **kwargs: pytest.fail("IBKR fallback should be disabled"),
    )

    out = provider.fetch_stock_bars(
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
    monkeypatch.setattr(provider, "load_stock_1m_from_lake", lambda *args, **kwargs: pd.DataFrame())

    monkeypatch.setattr(
        provider,
        "_fetch_ibkr_intraday",
        lambda *args, **kwargs: pytest.fail("IBKR fallback should be disabled"),
    )

    out = provider.fetch_stock_bars(
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
    monkeypatch.setattr(provider, "load_stock_1m_from_lake", lambda *args, **kwargs: _bars("2026-05-19 09:30", 2))
    ibkr = _bars("2026-05-19 10:00", 10)

    monkeypatch.setattr(provider, "_fetch_ibkr_intraday", lambda *args, **kwargs: ibkr)

    out = provider.fetch_stock_bars(
        "SPY",
        duration="30 D",
        bar_size="1 min",
        data_root=tmp_path,
    )

    pd.testing.assert_frame_equal(out, ibkr)
