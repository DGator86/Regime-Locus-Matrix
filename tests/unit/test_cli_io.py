"""Unit tests for rlm.cli.io — path resolution and load helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rlm.cli.io import (
    resolve_bars_path,
    resolve_chain_path,
    resolve_output_path,
    load_bars_dataframe,
    load_option_chain_dataframe,
)


class TestResolveBarsPath:
    def test_explicit_bars_arg_returned_as_is(self, tmp_path: Path) -> None:
        p = str(tmp_path / "custom_bars.csv")
        result = resolve_bars_path("SPY", bars_arg=p, data_root=None)
        assert result == Path(p).resolve()

    def test_no_arg_builds_default_path(self, tmp_path: Path) -> None:
        result = resolve_bars_path("SPY", bars_arg=None, data_root=str(tmp_path))
        assert result == tmp_path.resolve() / "raw" / "bars_SPY.csv"

    def test_symbol_used_verbatim(self, tmp_path: Path) -> None:
        result = resolve_bars_path("qqq", bars_arg=None, data_root=str(tmp_path))
        assert result.name == "bars_qqq.csv"


class TestResolveChainPath:
    def test_explicit_arg_returned_when_provided(self, tmp_path: Path) -> None:
        p = str(tmp_path / "chain.csv")
        result = resolve_chain_path("SPY", chain_arg=p, data_root=None)
        assert result == Path(p).resolve()

    def test_none_returned_when_file_absent(self, tmp_path: Path) -> None:
        result = resolve_chain_path("SPY", chain_arg=None, data_root=str(tmp_path))
        assert result is None

    def test_path_returned_when_file_exists(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        chain = raw / "option_chain_SPY.csv"
        chain.write_text("expiry,strike\n2024-01-19,450\n")
        result = resolve_chain_path("SPY", chain_arg=None, data_root=str(tmp_path))
        assert result == chain.resolve()


class TestResolveOutputPath:
    def test_explicit_file_path_returned(self, tmp_path: Path) -> None:
        p = str(tmp_path / "out.csv")
        result = resolve_output_path("forecast_features", "SPY", out_arg=p, data_root=None)
        assert result == Path(p).resolve()

    def test_explicit_directory_gets_filename(self, tmp_path: Path) -> None:
        result = resolve_output_path("forecast_features", "SPY", out_arg=str(tmp_path), data_root=None)
        assert result.name == "forecast_features_SPY.csv"
        assert result.parent == tmp_path.resolve()

    def test_default_path_under_processed(self, tmp_path: Path) -> None:
        result = resolve_output_path("forecast_features", "SPY", out_arg=None, data_root=str(tmp_path))
        assert result == tmp_path.resolve() / "processed" / "forecast_features_SPY.csv"


class TestLoadHelpers:
    def _write_bars_csv(self, path: Path) -> None:
        path.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01,100,105,99,104,1000000\n"
            "2024-01-02,104,108,103,107,900000\n"
        )

    def test_load_bars_dataframe_returns_dataframe(self, tmp_path: Path) -> None:
        csv = tmp_path / "bars_SPY.csv"
        self._write_bars_csv(csv)
        df = load_bars_dataframe(csv)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_option_chain_returns_none_for_missing(self, tmp_path: Path) -> None:
        result = load_option_chain_dataframe(tmp_path / "nonexistent.csv")
        assert result is None

    def test_load_option_chain_returns_dataframe_when_present(self, tmp_path: Path) -> None:
        csv = tmp_path / "option_chain_SPY.csv"
        csv.write_text("timestamp,expiry,strike,call_bid\n2024-01-01,2024-02-16,450,5.0\n")
        result = load_option_chain_dataframe(csv)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
