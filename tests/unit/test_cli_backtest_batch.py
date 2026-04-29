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
    )
    seen: list[tuple[str, str]] = []

    monkeypatch.setattr(backtest, "_parse_args", lambda: args)
    monkeypatch.setattr(backtest, "_resolve_symbols", lambda _args: ["SPY", "QQQ"])
    monkeypatch.setattr(backtest, "generate_run_id", lambda prefix: f"{prefix}-fixed")
    monkeypatch.setattr(
        backtest,
        "_run_one",
        lambda sym, _args, _out_dir, run_id: seen.append((sym, run_id)),
    )

    backtest.main()

    assert seen == [
        ("SPY", "backtest-fixed-01-SPY"),
        ("QQQ", "backtest-fixed-02-QQQ"),
    ]


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
