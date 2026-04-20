"""Unit tests for rlm.data.backend — backend resolution logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlm.data.backend import DataBackend, resolve_backend


class TestResolveBackendExplicit:
    def test_csv_string_returns_csv(self, tmp_path: Path) -> None:
        result = resolve_backend("csv", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.CSV

    def test_lake_string_returns_lake(self, tmp_path: Path) -> None:
        result = resolve_backend("lake", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.LAKE

    def test_enum_csv_accepted(self, tmp_path: Path) -> None:
        result = resolve_backend(DataBackend.CSV, "SPY", data_root=str(tmp_path))
        assert result == DataBackend.CSV

    def test_enum_lake_accepted(self, tmp_path: Path) -> None:
        result = resolve_backend(DataBackend.LAKE, "SPY", data_root=str(tmp_path))
        assert result == DataBackend.LAKE


class TestResolveBackendAuto:
    def test_auto_returns_csv_when_only_csv_exists(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bars_SPY.csv").write_text("timestamp,close\n2024-01-01,400\n")
        result = resolve_backend("auto", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.CSV

    def test_auto_returns_lake_when_lake_dir_exists(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake" / "SPY"
        lake.mkdir(parents=True)
        result = resolve_backend("auto", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.LAKE

    def test_auto_prefers_lake_over_csv(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bars_SPY.csv").write_text("timestamp,close\n")
        lake = tmp_path / "lake" / "SPY"
        lake.mkdir(parents=True)
        result = resolve_backend("auto", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.LAKE

    def test_auto_returns_csv_when_nothing_exists(self, tmp_path: Path) -> None:
        result = resolve_backend("auto", "SPY", data_root=str(tmp_path))
        assert result == DataBackend.CSV

    def test_auto_checks_interval_subdirectory(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake" / "SPY" / "1h"
        lake.mkdir(parents=True)
        result = resolve_backend("auto", "SPY", data_root=str(tmp_path), interval="1h")
        assert result == DataBackend.LAKE

    def test_auto_symbol_case_insensitive(self, tmp_path: Path) -> None:
        lake = tmp_path / "lake" / "SPY"
        lake.mkdir(parents=True)
        result = resolve_backend("auto", "spy", data_root=str(tmp_path))
        assert result == DataBackend.LAKE
