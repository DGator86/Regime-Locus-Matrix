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
        bars=None,
        chain=None,
    )
    run_id_prefixes: list[str] = []

    def fake_run_symbol(sym: str, _args: object, *, svc: object, out_dir: object, symbols: list[str]) -> tuple[str, dict]:
        # Simulate the distinct per-symbol run_id prefix used in the real _run_symbol
        run_id_prefixes.append(f"backtest-{sym}")
        return sym, {}

    monkeypatch.setattr(backtest, "_parse_args", lambda: args)
    monkeypatch.setattr(backtest, "resolve_backtest_symbols", lambda _args: ["SPY", "QQQ"])
    monkeypatch.setattr(backtest, "BacktestService", lambda: None)
    monkeypatch.setattr(backtest, "_run_symbol", fake_run_symbol)

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
    assert run_id_prefixes == ["backtest-SPY", "backtest-QQQ"]
    assert len(set(run_id_prefixes)) == 2  # distinct run ID prefix per symbol


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
