from __future__ import annotations

import csv
from pathlib import Path

from rlm.execution.trade_log_io import (
    open_plan_ids,
    seed_paper_opens_from_active_plans,
    trade_log_row_from_active_plan,
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


def test_seed_paper_open_accepts_credit_actives(tmp_path: Path) -> None:
    """Credits use entry_debit < 0; they must still open a monitorable paper row."""
    log_path = tmp_path / "options_large_account_trade_log.csv"
    plan = {
        "status": "active",
        "plan_id": "QQQ_20260804_1030",
        "symbol": "QQQ",
        "entry_debit_dollars": -120.0,
        "entry_mid_mark_dollars": -115.0,
        "decision": {"strategy_name": "long_call_spread"},
        "matched_legs": [
            {
                "side": "long",
                "option_type": "call",
                "strike": 480.0,
                "expiry": "2026-08-22",
                "mid": 4.5,
            },
            {
                "side": "short",
                "option_type": "call",
                "strike": 485.0,
                "expiry": "2026-08-22",
                "mid": 5.65,
            },
        ],
    }
    row = trade_log_row_from_active_plan(plan)
    assert row is not None
    assert float(row["entry_debit"]) == -120.0
    assert float(row["entry_mid"]) == -115.0
    assert float(row["current_mark"]) == -115.0
    assert float(row["unrealized_pnl"]) == 5.0

    seeded = seed_paper_opens_from_active_plans([plan], log_path)
    assert seeded == ["QQQ_20260804_1030"]
    assert "QQQ_20260804_1030" in open_plan_ids(log_path)
    with log_path.open(encoding="utf-8", newline="") as f:
        stored = next(csv.DictReader(f))
    assert stored["closed"] == "0"
    assert float(stored["entry_debit"]) == -120.0


def test_seed_paper_rejects_zero_cost_plans() -> None:
    assert (
        trade_log_row_from_active_plan(
            {
                "status": "active",
                "plan_id": "ZERO_1",
                "symbol": "SPY",
                "entry_debit_dollars": 0.0,
                "entry_mid_mark_dollars": 0.0,
            }
        )
        is None
    )
