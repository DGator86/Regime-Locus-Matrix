"""EOD PnL report: session bucketing, open vs closed, symbol drill-down."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.notify.pnl_report import calculate_daily_pnl  # noqa: E402


def test_eod_report_open_closed_and_worst_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    # Session date matches fake "now" below (20:00 UTC = 4pm Eastern same calendar day in EDT)
    rows = [
        f"{_ts(2026, 4, 24, 18)},p1,SPY,x,2,1,0.5,0.5,-1.5,-1,hold,0,1\n",  # open, red
        f"{_ts(2026, 4, 24, 19)},p2,QQQ,x,1,1,0.2,0.2,-0.8,-1,hold,0,1\n",  # open, red
        f"{_ts(2026, 4, 24, 19, 30)},p2,QQQ,x,1,1,0.2,0.2,-0.8,-1,hold,0,1\n",  # duplicate poll
        f"{_ts(2026, 4, 24, 19, 45)},p3,IWM,x,1,1,1.0,1.0,0.0,0,hard_stop,1,0\n",  # closed, flat
    ]
    (dproc / "trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    # 2026-04-24 20:00 UTC → 2026-04-24 Eastern (EDT)
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)

    text = calculate_daily_pnl(tmp_path)
    assert "2026-04-24" in text
    assert "Options" in text
    assert "Open: 2" in text
    assert "Exits (closed=1): 1" in text
    assert "IWM" in text or "Worst" in text
    # Missing parallel books: placeholders, not a failure
    assert "Equities" in text
    assert "no file" in text.lower() or "file empty" in text
    assert "Challenge" in text
    assert "no data/challenge/state" in text or "state.json" in text


def _ts(y: int, m: int, d: int, hour: int, minute: int = 0) -> str:
    return datetime(y, m, d, hour, minute, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_small_session_exits_line_has_newline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exits line must end with newline even when n_plan <= 30 (mtm_note suppressed)."""
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    rows = [
        f"{_ts(2026, 4, 24, 18)},p1,SPY,x,2,1,0.5,0.5,-1.5,-1,hold,1,1\n",  # closed, loss
    ]
    (dproc / "trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)

    text = calculate_daily_pnl(tmp_path)
    # "unique plan_id" must appear on its own line, not concatenated with exits line
    assert "\n  unique plan_id:" in text


def test_exit_payoff_ratio_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Payoff ratio line appears when there are both wins and losses in closed exits."""
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    rows = [
        f"{_ts(2026, 4, 24, 18)},win1,SPY,x,1,1,1.0,1.0,200.0,2,tp,1,0\n",  # closed win +200
        f"{_ts(2026, 4, 24, 18)},lose1,QQQ,x,1,1,0.5,0.5,-100.0,-1,stop,1,0\n",  # closed loss -100
        f"{_ts(2026, 4, 24, 18)},lose2,IWM,x,1,1,0.5,0.5,-50.0,-1,stop,1,0\n",  # closed loss -50
    ]
    (dproc / "trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)

    text = calculate_daily_pnl(tmp_path)
    assert "Exit payoff" in text
    assert "2.67x" in text  # avg win 200 / avg loss 75 = 2.666...


def test_concentration_warning_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concentration line fires when one symbol >= 40% of total session loss."""
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    rows = [
        f"{_ts(2026, 4, 24, 18)},m1,META,x,1,1,0.5,0.5,-500.0,-1,hold,0,5\n",  # META big loss
        f"{_ts(2026, 4, 24, 18)},s1,SPY,x,1,1,0.9,0.9,-100.0,-1,hold,0,5\n",  # SPY small loss
    ]
    (dproc / "trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)

    text = calculate_daily_pnl(tmp_path)
    assert "Concentration" in text
    assert "META" in text
    assert "83%" in text  # 500 / 600 = 83.3%


def test_concentration_warning_not_shown_when_spread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No concentration warning when no single symbol dominates."""
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    rows = [
        f"{_ts(2026, 4, 24, 18)},a1,AAPL,x,1,1,0.7,0.7,-100.0,-1,hold,0,5\n",
        f"{_ts(2026, 4, 24, 18)},m1,MSFT,x,1,1,0.7,0.7,-120.0,-1,hold,0,5\n",
        f"{_ts(2026, 4, 24, 18)},g1,GOOGL,x,1,1,0.7,0.7,-110.0,-1,hold,0,5\n",
    ]
    (dproc / "trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)

    text = calculate_daily_pnl(tmp_path)
    assert "Concentration" not in text


def test_eod_includes_challenge_when_state_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,"
        "unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n",
        encoding="utf-8",
    )
    ch = tmp_path / "data" / "challenge"
    ch.mkdir(parents=True)
    ch_state = (
        '{"balance": 1100, "seed": 1000, "target": 25000, '
        '"open_positions": [], "trade_history": []}'
    )
    (ch / "state.json").write_text(ch_state, encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)
    out = calculate_daily_pnl(tmp_path)
    assert "Challenge" in out
    assert "1,100.00" in out
