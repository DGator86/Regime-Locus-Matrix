"""Backtest CLI symbol resolution and expanded universe."""

import argparse

import pytest

from rlm.cli.common import resolve_backtest_symbols
from rlm.data.liquidity_universe import (
    EXPANDED_LIQUID_UNIVERSE,
    LIQUID_STOCK_EXTRAS,
    LIQUID_UNIVERSE,
)


def test_expanded_liquid_universe_is_deduped_union() -> None:
    assert len(EXPANDED_LIQUID_UNIVERSE) == len(LIQUID_UNIVERSE) + len(LIQUID_STOCK_EXTRAS)
    assert set(EXPANDED_LIQUID_UNIVERSE) == set(LIQUID_UNIVERSE) | set(LIQUID_STOCK_EXTRAS)


def test_resolve_backtest_symbols_single_and_csv() -> None:
    ns = argparse.Namespace(symbol="XOM", symbols=None, universe=False)
    assert resolve_backtest_symbols(ns) == ["XOM"]
    ns2 = argparse.Namespace(symbol="SPY", symbols="qqq, aapl", universe=False)
    assert resolve_backtest_symbols(ns2) == ["QQQ", "AAPL"]


def test_resolve_universe_uses_expanded() -> None:
    ns = argparse.Namespace(symbol="SPY", symbols=None, universe=True)
    assert resolve_backtest_symbols(ns) == list(EXPANDED_LIQUID_UNIVERSE)


def test_resolve_both_universe_and_symbols_errors() -> None:
    ns = argparse.Namespace(symbol="SPY", symbols="SPY,QQQ", universe=True)
    with pytest.raises(SystemExit):
        resolve_backtest_symbols(ns)
