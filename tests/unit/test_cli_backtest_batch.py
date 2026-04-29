from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
from types import SimpleNamespace

from rlm.cli import backtest


@dataclass
class _FakeConfig:
    regime_model: str = "hmm"


class _FakeService:
    def run(self, req):
        return object()

    def write_outputs(self, req, result):
        return SimpleNamespace(
            trades_csv=None,
            equity_csv=None,
            trades_rows=0,
            equity_rows=0,
        )

    def summarize(self, result):
        return {"metrics": {"sharpe": 1.0}}


def test_multi_symbol_backtest_uses_distinct_run_ids(monkeypatch, tmp_path):
    args = SimpleNamespace(
        out_dir=str(tmp_path / "processed"),
        data_root=str(tmp_path),
        synthetic=False,
        bars=None,
        chain=None,
        backend="csv",
        profile=None,
        walkforward=False,
        initial_capital=100_000.0,
    )

    monkeypatch.setattr(backtest, "generate_run_id", lambda prefix: f"{prefix}-fixed")
    monkeypatch.setattr(
        backtest,
        "_load_symbol_data",
        lambda *_args, **_kwargs: (SimpleNamespace(), None),
    )
    monkeypatch.setattr(backtest, "build_pipeline_config", lambda *_args, **_kwargs: _FakeConfig())

    symbols = ["SPY", "QQQ"]
    for sym in symbols:
        backtest._run_symbol(
            sym,
            args,
            svc=_FakeService(),
            out_dir=tmp_path / "processed",
            symbols=symbols,
        )

    runs_dir = tmp_path / "artifacts" / "runs"
    assert (runs_dir / "backtest-SPY-fixed.json").exists()
    assert (runs_dir / "backtest-QQQ-fixed.json").exists()


def test_run_walkforward_wrapper_allows_single_symbol_override():
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [sys.executable, "scripts/run_walkforward.py", "--symbol", "SPY", "--help"],
        cwd=".",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "not allowed with argument --universe" not in result.stderr
    assert "--universe" in result.stdout
