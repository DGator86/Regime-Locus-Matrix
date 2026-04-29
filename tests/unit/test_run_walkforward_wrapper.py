"""Legacy walk-forward wrapper argument rewriting."""

from __future__ import annotations

import runpy
import sys

from rlm.cli.backtest import _parse_args
from rlm.cli.common import resolve_backtest_symbols


def _load_wrapper_with_args(monkeypatch, args: list[str]) -> list[str]:
    monkeypatch.setattr(sys, "argv", ["scripts/run_walkforward.py", *args])
    runpy.run_path("scripts/run_walkforward.py", run_name="not_main")
    return list(sys.argv)


def test_run_walkforward_defaults_to_universe(monkeypatch) -> None:
    argv = _load_wrapper_with_args(monkeypatch, [])

    assert argv == ["scripts/run_walkforward.py", "--walkforward", "--universe"]


def test_run_walkforward_preserves_single_symbol_override(monkeypatch) -> None:
    _load_wrapper_with_args(monkeypatch, ["--symbol", "SPY"])

    args = _parse_args()
    assert args.walkforward is True
    assert args.universe is False
    assert resolve_backtest_symbols(args) == ["SPY"]


def test_run_walkforward_preserves_symbols_override(monkeypatch) -> None:
    _load_wrapper_with_args(monkeypatch, ["--symbols", "SPY,QQQ"])

    args = _parse_args()
    assert args.walkforward is True
    assert args.universe is False
    assert resolve_backtest_symbols(args) == ["SPY", "QQQ"]
