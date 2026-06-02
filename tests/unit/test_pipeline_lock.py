"""Tests for universe pipeline single-instance lock."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fcntl pipeline lock is Linux-only")

from rlm.utils.pipeline_lock import PipelineLockError, universe_pipeline_lock


def test_universe_pipeline_lock_blocks_second_holder(tmp_path: Path) -> None:
    with universe_pipeline_lock(tmp_path):
        with pytest.raises(PipelineLockError):
            with universe_pipeline_lock(tmp_path):
                pass


def test_universe_pipeline_lock_released_after_context(tmp_path: Path) -> None:
    with universe_pipeline_lock(tmp_path):
        pass
    with universe_pipeline_lock(tmp_path):
        pass


def test_cli_skips_when_lock_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("RLM_ROOT", str(tmp_path))
    with universe_pipeline_lock(tmp_path):
        proc = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts" / "run_universe_options_pipeline.py"),
                "--symbols",
                "SPY",
                "--out",
                "data/processed/universe_trade_plans.json",
                "--top",
                "1",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert proc.returncode == 0
    assert "pipeline-lock" in proc.stdout + proc.stderr
