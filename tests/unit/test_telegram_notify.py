"""Telegram notify state machine (seeding, no real API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.notify.telegram_rlm import notification_cycle  # noqa: E402


def test_notify_seed_silent(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(
        json.dumps({"results": [{"status": "active", "plan_id": "p1", "symbol": "SPY", "strategy": "s"}]}),
        encoding="utf-8",
    )
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2024-01-01T00:00:00Z,p1,SPY,x,1,1,1.1,1.1,0.1,5,hold,0,5\n",
        encoding="utf-8",
    )
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")

    msgs, blob = notification_cycle(
        tmp_path,
        {
            "notify_seeded": False,
        },
    )
    assert msgs == []
    assert blob.get("notify_seeded") is True
    assert "p1" in blob.get("known_option_plans", [])


def test_new_plan_after_seed(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "active",
                        "plan_id": "p1",
                        "symbol": "SPY",
                        "strategy": "a",
                        "entry_debit_dollars": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2024-01-01T00:00:00Z,p1,SPY,x,1,1,1,1,0,0,hold,0,5\n",
        encoding="utf-8",
    )
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    s0, b0 = notification_cycle(
        tmp_path,
        {
            "notify_seeded": False,
        },
    )
    assert s0 == [] and b0.get("notify_seeded") is True
    (dproc / "universe_trade_plans.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "status": "active",
                        "plan_id": "p1",
                        "symbol": "SPY",
                        "strategy": "a",
                    },
                    {
                        "status": "active",
                        "plan_id": "p2",
                        "symbol": "QQQ",
                        "strategy": "b",
                        "entry_debit_dollars": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    s1, _ = notification_cycle(tmp_path, b0)
    assert len(s1) == 1
    assert "p2" in s1[0] and "QQQ" in s1[0]
