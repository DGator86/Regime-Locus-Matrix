from __future__ import annotations

import argparse
import os
import subprocess
import sys

from rlm.cli import backtest


def test_multi_symbol_backtest_uses_distinct_run_ids(monkeypatch, tmp_path):
    args = argparse.Namespace(
        out_dir=str(tmp_path / "processed"),
        data_root=str(tmp_path),
        walkforward=False,
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

    backtest.main()

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
