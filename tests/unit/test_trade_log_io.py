from __future__ import annotations

import csv
from pathlib import Path

from rlm.execution.trade_log_io import (
    closed_plan_ids,
    open_plan_ids,
    seed_paper_opens_from_active_plans,
    upsert_trade_log_row,
)


def test_seed_paper_open_matches_plan_debit(tmp_path: Path) -> None:
    log_path = tmp_path / "trade_log.csv"
    plan = {
        "status": "active",
        "plan_id": "SPY_20260603_1500",
        "symbol": "SPY",
        "entry_debit_dollars": 150.0,
        "entry_mid_mark_dollars": 150.0,
        "decision": {"strategy_name": "scalp_long_straddle"},
        "matched_legs": [
            {
                "side": "long",
                "option_type": "call",
                "strike": 600.0,
                "expiry": "2026-06-06",
            }
        ],
        "thresholds": {"v_take_profit": 200.0, "v_hard_stop": 80.0},
    }
    seeded = seed_paper_opens_from_active_plans([plan], log_path)
    assert seeded == ["SPY_20260603_1500"]
    assert "SPY_20260603_1500" in open_plan_ids(log_path)
    text = log_path.read_text(encoding="utf-8")
    assert "150.0" in text
    assert seed_paper_opens_from_active_plans([plan], log_path) == []


def test_seed_paper_does_not_reopen_closed_plan_id(tmp_path: Path) -> None:
    """Monitor close + paper reseed against the same actives must not revive the row."""
    log_path = tmp_path / "options_large_account_trade_log.csv"
    plan = {
        "status": "active",
        "plan_id": "NVDA_20260808_1500",
        "symbol": "NVDA",
        "entry_debit_dollars": 2.0,
        "entry_mid_mark_dollars": 2.0,
        "decision": {"strategy_name": "long_call_spread"},
        "matched_legs": [],
    }
    assert seed_paper_opens_from_active_plans([plan], log_path) == ["NVDA_20260808_1500"]
    upsert_trade_log_row(
        log_path,
        {
            "plan_id": "NVDA_20260808_1500",
            "closed": "1",
            "signal": "take_profit",
            "current_mark": "3.0",
            "peak_mark": "3.0",
        },
    )
    assert open_plan_ids(log_path) == set()
    assert "NVDA_20260808_1500" in closed_plan_ids(log_path)

    assert seed_paper_opens_from_active_plans([plan], log_path) == []
    assert open_plan_ids(log_path) == set()

    with log_path.open(encoding="utf-8", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r.get("plan_id") == "NVDA_20260808_1500")
    assert row["closed"] == "1"
    assert row["signal"] == "take_profit"
    assert row["peak_mark"] == "3.0"


def test_upsert_does_not_reopen_closed_plan_id(tmp_path: Path) -> None:
    """Sticky closed: concurrent seed/monitor hold must not revive an exited row."""
    log_path = tmp_path / "options_large_account_trade_log.csv"
    upsert_trade_log_row(
        log_path,
        {
            "plan_id": "AAPL_20260808_1200",
            "symbol": "AAPL",
            "signal": "take_profit",
            "closed": "1",
            "peak_mark": "4.5",
            "current_mark": "4.5",
            "entry_debit": "2.0",
            "entry_mid": "2.0",
        },
    )
    assert "AAPL_20260808_1200" in closed_plan_ids(log_path)

    upsert_trade_log_row(
        log_path,
        {
            "plan_id": "AAPL_20260808_1200",
            "symbol": "AAPL",
            "signal": "hold",
            "closed": "0",
            "peak_mark": "2.0",
            "current_mark": "2.0",
            "entry_debit": "2.0",
            "entry_mid": "2.0",
        },
    )
    assert open_plan_ids(log_path) == set()
    with log_path.open(encoding="utf-8", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r.get("plan_id") == "AAPL_20260808_1200")
    assert row["closed"] == "1"
    assert row["signal"] == "take_profit"
    assert row["peak_mark"] == "4.5"
