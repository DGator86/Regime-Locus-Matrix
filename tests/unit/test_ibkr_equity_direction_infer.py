"""Infer bull/bear for equity leg from universe plan rows (regime_key / strategy)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ibkr_equity_paper_trade as mod  # noqa: E402


def test_infer_from_regime_key_head() -> None:
    assert mod._infer_regime_direction({"regime_key": "bull|mid|liq|flow"}) == "bull"  # noqa: SLF001
    assert mod._infer_regime_direction({"regime_key": "bear|mid|liq|flow"}) == "bear"  # noqa: SLF001


def test_infer_explicit_regime_direction_wins() -> None:
    p = {"regime_direction": "bear", "regime_key": "bull|x", "strategy_name": "debit_spread_call"}
    assert mod._infer_regime_direction(p) == "bear"  # noqa: SLF001


def test_infer_from_debit_spread_strategy() -> None:
    assert mod._infer_regime_direction({"strategy_name": "debit_spread_call"}) == "bull"  # noqa: SLF001
    assert mod._infer_regime_direction({"strategy_name": "debit_spread_put"}) == "bear"  # noqa: SLF001
    assert mod._infer_regime_direction({"decision": {"strategy_name": "bull_call_spread"}}) == "bull"  # noqa: SLF001


def test_infer_empty_when_unknown() -> None:
    assert mod._infer_regime_direction({}) == ""  # noqa: SLF001
    assert mod._infer_regime_direction({"regime_key": "range|x|y|z"}) == ""  # noqa: SLF001
