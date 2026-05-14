"""Telegram notify state machine (seeding, no real API)."""

from __future__ import annotations

import csv
import io
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
    assert any("LARGE OPTIONS" in m and "NEW POSITION" in m and "p2" in m and "QQQ" in m for m in s1)


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
    assert "LARGE OPTIONS" in s2[0] and "CLOSED" in s2[0]


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


def test_challenge_positions_show_occ_and_state_freshness(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    ch = tmp_path / "data" / "challenge"
    ch.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n",
        encoding="utf-8",
    )
    state = {
        "balance": 500,
        "seed": 1000,
        "target": 25000,
        "session_count": 3,
        "created_at": "2026-05-01T00:00:00Z",
        "last_updated": "2026-05-13T16:00:00Z",
        "open_positions": [
            {
                "position_id": "abc12345",
                "symbol": "SPY",
                "option_type": "call",
                "direction": "long",
                "underlying_entry": 500.0,
                "strike": 505.0,
                "dte_at_entry": 14,
                "entry_date": "2026-05-01",
                "premium_per_share": 2.5,
                "qty": 1,
                "total_cost": 250.0,
                "delta_at_entry": 0.35,
                "iv_at_entry": 0.2,
                "dte_remaining": 10,
                "current_premium": 2.7,
                "current_value": 270.0,
                "unrealised_pnl": 20.0,
                "status": "open",
            }
        ],
        "trade_history": [],
    }
    (ch / "state.json").write_text(json.dumps(state), encoding="utf-8")
    text = build_universe_and_positions(tmp_path, max_positions=10)
    assert "SPY $505 Call - Exp. 05.15.26" in text
    assert "Cost - $250.00" in text
    assert "Current val - $270.00" in text
    assert "Current PnL - $20.00" in text
    assert "last_updated=2026-05-13T16:00:00Z" in text
    assert "state.json mtime" in text
    assert "challenge session runs" in text


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


def test_large_options_positions_human_format(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    plans = {
        "results": [
            {
                "plan_id": "plan_a",
                "symbol": "SPY",
                "strategy": "debit_spread_call",
                "decision": {"strategy_name": "debit_spread_call"},
                "matched_legs": [
                    {
                        "side": "long",
                        "option_type": "call",
                        "strike": 719.0,
                        "expiry": "2026-05-20",
                    }
                ],
            }
        ]
    }
    (dproc / "universe_trade_plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2026-04-28T00:00:00Z,plan_a,SPY,x,123.45,120,130.25,130.25,6.8,5.5,hold,0,12\n",
        encoding="utf-8",
    )
    text = build_universe_and_positions(tmp_path, max_positions=10)
    assert "SPY $719 Call - Exp. 05.20.26" in text
    assert "Cost - $123.45" in text
    assert "Current val - $130.25" in text
    assert "Current PnL - $6.80" in text
    assert "plan_a" in text


def test_large_options_uses_trade_plan_snapshots_when_missing_from_universe(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "universe_trade_plans.json").write_text(json.dumps({"results": []}), encoding="utf-8")
    snap = {
        "orphan_pid": {
            "plan_id": "orphan_pid",
            "symbol": "AAPL",
            "strategy": "debit_spread_call",
            "matched_legs": [
                {
                    "side": "long",
                    "option_type": "call",
                    "strike": 190.0,
                    "expiry": "2026-06-13",
                }
            ],
        }
    }
    (dproc / "trade_plan_snapshots.json").write_text(json.dumps(snap), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
        "2026-04-28T00:00:00Z,orphan_pid,AAPL,x,100,100,105,105,5,5,hold,0,20\n",
        encoding="utf-8",
    )
    text = build_universe_and_positions(tmp_path, max_positions=10)
    assert "AAPL $190 Call - Exp. 06.13.26" in text
    assert "orphan_pid" in text
    assert "Cost - $100.00" in text
    assert "Current val - $105.00" in text
    assert "Current PnL - $5.00" in text


def test_large_options_uses_trade_log_legs_json_when_universe_has_no_legs(tmp_path: Path) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    plans = {
        "results": [
            {
                "plan_id": "plan_b",
                "symbol": "META",
                "strategy": "debit_spread_put",
                "decision": {"strategy_name": "debit_spread_put"},
            }
        ]
    }
    (dproc / "universe_trade_plans.json").write_text(json.dumps(plans), encoding="utf-8")
    (dproc / "equity_positions_state.json").write_text("{}", encoding="utf-8")
    legs = [
        {"side": "long", "option_type": "put", "strike": 600.0, "expiry": "2026-06-05"},
        {"side": "short", "option_type": "put", "strike": 590.0, "expiry": "2026-06-05"},
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "timestamp_utc",
            "plan_id",
            "symbol",
            "strategy",
            "entry_debit",
            "entry_mid",
            "current_mark",
            "peak_mark",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "signal",
            "closed",
            "dte",
            "legs_json",
        ]
    )
    w.writerow(
        [
            "2026-04-28T00:00:00Z",
            "plan_b",
            "META",
            "x",
            "410",
            "410",
            "385",
            "385",
            "-25",
            "-6",
            "hold",
            "0",
            "20",
            json.dumps(legs),
        ]
    )
    (dproc / "trade_log.csv").write_text(buf.getvalue(), encoding="utf-8")
    text = build_universe_and_positions(tmp_path, max_positions=10)
    assert "META (Put debit spread" in text or "Put debit spread" in text
    assert "$600 Put" in text and "$590 Put" in text
    assert "plan_b" in text
