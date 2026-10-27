"""Crash safety for the 1m stock lake merge used by the EODHD collector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rlm.data.lake import stock_1m_parquet
from rlm.data.stock_bars_provider import load_stock_1m_from_lake, merge_bars_into_lake


def _bars(ts: str, close: float) -> pd.DataFrame:
    stamp = pd.Timestamp(ts)
    return pd.DataFrame(
        {
            "timestamp": [stamp],
            "open": [close],
            "high": [close + 0.5],
            "low": [close - 0.5],
            "close": [close],
            "volume": [1_000.0],
            "vwap": [close],
        }
    )


def test_merge_bars_into_lake_keeps_prior_file_when_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed publish must not truncate the symbol lake the collector rewrites."""
    merge_bars_into_lake("SPY", _bars("2026-05-22 15:55:00", 745.2), root=tmp_path)
    lake_path = stock_1m_parquet("SPY", duration_slug="full", root=tmp_path)
    original = lake_path.read_bytes()

    def _fail_replace(src: str, dst: str) -> None:
        raise OSError("publish interrupted")

    monkeypatch.setattr("rlm.data.lake.writers.os.replace", _fail_replace)

    with pytest.raises(OSError, match="publish interrupted"):
        merge_bars_into_lake("SPY", _bars("2026-05-22 15:56:00", 746.0), root=tmp_path)

    assert lake_path.read_bytes() == original
    loaded = load_stock_1m_from_lake("SPY", root=tmp_path)
    assert len(loaded) == 1
    assert float(loaded.iloc[-1]["close"]) == pytest.approx(745.2)
    assert not list(lake_path.parent.glob("*.tmp"))


def test_merge_bars_into_lake_replaces_with_merged_rows(tmp_path: Path) -> None:
    merge_bars_into_lake("SPY", _bars("2026-05-22 15:55:00", 745.2), root=tmp_path)
    merge_bars_into_lake("SPY", _bars("2026-05-22 15:56:00", 746.0), root=tmp_path)
    loaded = load_stock_1m_from_lake("SPY", root=tmp_path)
    assert list(loaded["close"]) == pytest.approx([745.2, 746.0])
