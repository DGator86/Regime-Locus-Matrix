"""Tests for plain-language option descriptions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.notify.options_plain_language import (  # noqa: E402
    explain_trade_plan,
    humanize_strategy_name,
    strategy_what_you_want,
)


def test_humanize_debit_spread_put() -> None:
    assert "Put debit spread" in humanize_strategy_name("debit_spread_put")


def test_strategy_what_you_want_bullish_spread() -> None:
    t = strategy_what_you_want("debit_spread_call", "SPY")
    assert "SPY" in t and "rise" in t.lower()


def test_explain_trade_plan_includes_legs() -> None:
    plan = {
        "plan_id": "x1",
        "symbol": "META",
        "strategy": "debit_spread_put",
        "decision": {"strategy_name": "debit_spread_put"},
        "matched_legs": [
            {"side": "long", "option_type": "put", "strike": 600, "expiry": "2026-06-05"},
            {"side": "short", "option_type": "put", "strike": 590, "expiry": "2026-06-05"},
        ],
        "entry_debit_dollars": 410.0,
        "thresholds": {"v_take_profit": 6.2, "v_hard_stop": 1.1},
    }
    txt = explain_trade_plan(plan)
    assert "x1" in txt and "META" in txt
    assert "Buy" in txt and "$600" in txt
    assert "Take-profit" in txt


def test_translate_script_single_plan(tmp_path: Path) -> None:
    plan = {
        "plan_id": "p",
        "symbol": "AAPL",
        "decision": {"strategy_name": "long_call"},
        "matched_legs": [{"side": "long", "option_type": "call", "strike": 200, "expiry": "20260615"}],
    }
    p = tmp_path / "one.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "translate_options_plain_language.py"), "--single-plan-json", str(p)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "AAPL" in r.stdout and "call" in r.stdout.lower()
