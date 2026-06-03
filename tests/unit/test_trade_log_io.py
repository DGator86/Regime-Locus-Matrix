from __future__ import annotations

from pathlib import Path

from rlm.execution.trade_log_io import open_plan_ids, seed_paper_opens_from_active_plans


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
