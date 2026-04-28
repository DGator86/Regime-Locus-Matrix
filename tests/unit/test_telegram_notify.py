"""Telegram notify state machine (seeding, no real API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.notify.telegram_rlm import build_universe_and_positions, notification_cycle  # noqa: E402


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
    assert "p1" in blob.get("announced_trade_open", [])
    assert "p1" in blob.get("last_universe_active_ids", [])


def test_new_position_after_seed(tmp_path: Path) -> None:
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
    log_lines = [
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n",
        "2024-01-01T00:00:00Z,p1,SPY,x,1,1,1,1,0,0,hold,0,5\n",
        "2024-01-01T00:00:01Z,p2,QQQ,x,1,1,1.05,1.05,0,0,hold,0,5\n",
    ]
    (dproc / "trade_log.csv").write_text("".join(log_lines), encoding="utf-8")
    s1, _ = notification_cycle(tmp_path, b0)
    # Universe "new idea" alerts are disabled to reduce chatter; trade_log still notifies.
    assert len(s1) == 1
    assert any("New position" in m and "p2" in m and "QQQ" in m for m in s1)


def test_profit_target_and_exit_alerts(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    header = "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    (dproc / "trade_log.csv").write_text(
        header + "2024-01-01T00:00:00Z,p9,SPY,x,1,1,1,1,0,0,hold,0,5\n",
        encoding="utf-8",
    )
    s0, b0 = notification_cycle(tmp_path, {"notify_seeded": False})
    assert s0 == [] and b0.get("notify_seeded") is True
    (dproc / "trade_log.csv").write_text(
        header + "2024-01-01T00:01:00Z,p9,SPY,x,1,1,1.2,1.2,0,0,take_profit,0,5\n",
        encoding="utf-8",
    )
    s1, b1 = notification_cycle(tmp_path, b0)
    assert len(s1) == 1
    assert "profit target" in s1[0].lower()
    (dproc / "trade_log.csv").write_text(
        header + "2024-01-01T00:02:00Z,p9,SPY,x,1,1,1.2,1.2,0,0,take_profit,1,5\n",
        encoding="utf-8",
    )
    s2, _ = notification_cycle(tmp_path, b1)
    assert len(s2) == 1
    assert "Exited position" in s2[0]


def test_legacy_state_migrates_announced_trade_open(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2024-01-01T00:00:00Z,p1,SPY,x,1,1,1,1,0,0,hold,0,5\n",
        encoding="utf-8",
    )
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    legacy = {
        "notify_seeded": True,
        "known_option_plans": ["p1"],
        "last_opt_signal": {"p1": "hold"},
        "announced_tp": [],
        "announced_exit": [],
    }
    s, b = notification_cycle(tmp_path, legacy)
    assert s == []
    assert "p1" in b.get("announced_trade_open", [])


def test_portfolio_report_flags_risk_warnings(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2026-04-28T00:00:00Z,p1,TSLA,x,1,1,1,1,-0.75,-75,hold,0,30\n"
        "2026-04-28T00:00:00Z,p2,NVDA,x,1,1,1,1,0.0,0,hold,0,20\n"
        "2026-04-28T00:00:00Z,p3,SPY,x,1,1,1,1,0.5,50,hold,0,13\n",
        encoding="utf-8",
    )
    text = build_universe_and_positions(tmp_path, max_positions=10)
    assert "⚠ MAX_LOSS_BREACH" in text
    assert "⚠ TIME_STOP_ZONE" in text
    assert "⚠ FORCE_CLOSE_ZONE" in text
