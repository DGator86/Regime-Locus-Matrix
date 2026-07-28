from __future__ import annotations

from pathlib import Path

from rlm.execution.trade_log_io import open_plan_ids, open_symbols, seed_paper_opens_from_active_plans


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


def test_seed_skips_when_symbol_already_open_under_rotated_plan_id(tmp_path: Path) -> None:
    log_path = tmp_path / "trade_log.csv"
    first = {
        "status": "active",
        "plan_id": "AAPL_20260728_1000",
        "symbol": "AAPL",
        "entry_debit_dollars": 1.0,
        "entry_mid_mark_dollars": 1.0,
    }
    rotated = {
        "status": "active",
        "plan_id": "AAPL_20260728_1005",
        "symbol": "AAPL",
        "entry_debit_dollars": 1.2,
        "entry_mid_mark_dollars": 1.2,
    }
    assert seed_paper_opens_from_active_plans([first], log_path) == ["AAPL_20260728_1000"]
    assert open_symbols(log_path) == {"AAPL"}
    assert seed_paper_opens_from_active_plans([rotated], log_path) == []
    assert open_plan_ids(log_path) == {"AAPL_20260728_1000"}
