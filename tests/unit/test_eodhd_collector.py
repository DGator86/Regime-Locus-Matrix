"""EODHD collector scheduling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.eodhd_collector import default_poll_symbols, refresh_symbol_intraday
from rlm.data.lake import stock_1m_parquet


def test_default_poll_symbols_spy_qqq_first() -> None:
    syms = default_poll_symbols()
    assert syms[0] == "SPY"
    assert syms[1] == "QQQ"
    assert len(syms) == 12


def test_refresh_symbol_exports_chart_csv_when_no_incremental_bars(
    tmp_path: Path, monkeypatch
) -> None:
    """Dashboard chart export must run even when EODHD returns no new bars."""
    sym = "SPY"
    lake_path = stock_1m_parquet(sym, duration_slug="full", root=tmp_path)
    lake_path.parent.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp("2026-05-22 19:55:00")
    pd.DataFrame(
        {
            "timestamp": [ts],
            "open": [745.0],
            "high": [745.5],
            "low": [744.5],
            "close": [745.2],
            "volume": [1000.0],
            "vwap": [745.1],
        }
    ).to_parquet(lake_path, index=False)

    fixed_now = pd.Timestamp("2026-05-22 20:00:00", tz="America/New_York")

    class _FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr("rlm.data.eodhd_collector.datetime", _FixedDatetime)
    monkeypatch.setattr(
        "rlm.data.eodhd_collector.fetch_intraday_1m",
        lambda *a, **k: pd.DataFrame(),
    )

    refresh_symbol_intraday(sym, root=tmp_path, api_key="test-key")
    chart = tmp_path / "data" / "processed" / "chart_bars_spy.csv"
    assert chart.is_file()
    out = pd.read_csv(chart)
    assert len(out) == 1
    assert float(out.iloc[-1]["close"]) == 745.2
