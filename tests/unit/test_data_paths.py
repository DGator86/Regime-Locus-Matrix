"""Unit tests for rlm.data.paths — path resolution contract."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rlm.data.paths import (
    get_artifacts_dir,
    get_data_root,
    get_models_dir,
    get_processed_data_dir,
    get_raw_data_dir,
)


class TestGetDataRoot:
    def test_explicit_arg_takes_priority(self, tmp_path: Path) -> None:
        result = get_data_root(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_explicit_path_object_accepted(self, tmp_path: Path) -> None:
        result = get_data_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_env_var_used_when_no_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RLM_DATA_ROOT", str(tmp_path))
        result = get_data_root()
        assert result == tmp_path.resolve()

    def test_explicit_overrides_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        other = tmp_path / "other"
        monkeypatch.setenv("RLM_DATA_ROOT", str(tmp_path))
        result = get_data_root(str(other))
        assert result == other.resolve()

    def test_fallback_to_cwd_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RLM_DATA_ROOT", raising=False)
        result = get_data_root()
        assert result == (Path.cwd() / "data").resolve()

    def test_empty_env_var_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RLM_DATA_ROOT", "   ")
        result = get_data_root()
        assert result == (Path.cwd() / "data").resolve()

    def test_result_is_absolute(self, tmp_path: Path) -> None:
        result = get_data_root(str(tmp_path))
        assert result.is_absolute()


class TestSubdirectories:
    def test_raw_dir_is_child_of_root(self, tmp_path: Path) -> None:
        result = get_raw_data_dir(str(tmp_path))
        assert result == tmp_path.resolve() / "raw"

    def test_processed_dir_is_child_of_root(self, tmp_path: Path) -> None:
        result = get_processed_data_dir(str(tmp_path))
        assert result == tmp_path.resolve() / "processed"

    def test_processed_dir_created_automatically(self, tmp_path: Path) -> None:
        result = get_processed_data_dir(str(tmp_path))
        assert result.is_dir()

    def test_models_dir_is_child_of_root(self, tmp_path: Path) -> None:
        result = get_models_dir(str(tmp_path))
        assert result == tmp_path.resolve() / "models"

    def test_models_dir_created_automatically(self, tmp_path: Path) -> None:
        result = get_models_dir(str(tmp_path))
        assert result.is_dir()

    def test_artifacts_dir_is_child_of_root(self, tmp_path: Path) -> None:
        result = get_artifacts_dir(str(tmp_path))
        assert result == tmp_path.resolve() / "artifacts"

    def test_artifacts_dir_created_automatically(self, tmp_path: Path) -> None:
        result = get_artifacts_dir(str(tmp_path))
        assert result.is_dir()
