import pytest
from pathlib import Path

from rlm.data.paths import get_data_root, get_rlm_processed_dir, get_rlm_runtime_root


def test_data_root_override(tmp_path):
    assert get_data_root(tmp_path) == tmp_path.resolve()


def test_rlm_runtime_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLM_ROOT", str(tmp_path))
    assert get_rlm_runtime_root() == tmp_path.resolve()
    assert get_rlm_processed_dir() == tmp_path.resolve() / "data" / "processed"
