"""Unit tests for rlm.data.readers — CSV loading contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rlm.data.readers import load_bars, load_option_chain


def _write_bars(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,100,105,99,104,1000000\n"
        "2024-01-02,104,108,103,107,900000\n"
        "2024-01-03,107,110,106,109,800000\n"
    )


def _write_chain(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,expiry,strike,call_bid,call_ask\n"
        "2024-01-01,2024-02-16,450,5.0,5.5\n"
        "2024-01-01,2024-02-16,455,3.0,3.5\n"
    )


class TestLoadBarsCSV:
    def test_explicit_path_loads_successfully(self, tmp_path: Path) -> None:
        csv = tmp_path / "bars_SPY.csv"
        _write_bars(csv)
        df = load_bars("SPY", bars_path=csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_returns_timestamp_index(self, tmp_path: Path) -> None:
        csv = tmp_path / "bars_SPY.csv"
        _write_bars(csv)
        df = load_bars("SPY", bars_path=csv)
        assert df.index.name == "timestamp"

    def test_csv_backend_uses_raw_dir(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        csv = raw / "bars_SPY.csv"
        _write_bars(csv)
        df = load_bars("SPY", data_root=str(tmp_path), backend="csv")
        assert len(df) == 3

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="bars"):
            load_bars("MISSING", data_root=str(tmp_path), backend="csv")

    def test_symbol_uppercased_for_path_lookup(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        csv = raw / "bars_SPY.csv"
        _write_bars(csv)
        df = load_bars("spy", data_root=str(tmp_path), backend="csv")
        assert len(df) == 3


class TestLoadOptionChainCSV:
    def test_explicit_path_loads_successfully(self, tmp_path: Path) -> None:
        csv = tmp_path / "option_chain_SPY.csv"
        _write_chain(csv)
        df = load_option_chain("SPY", chain_path=csv)
        assert df is not None
        assert len(df) == 2

    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        result = load_option_chain("SPY", data_root=str(tmp_path), backend="csv")
        assert result is None

    def test_explicit_path_missing_returns_none(self, tmp_path: Path) -> None:
        result = load_option_chain("SPY", chain_path=tmp_path / "nope.csv")
        assert result is None
