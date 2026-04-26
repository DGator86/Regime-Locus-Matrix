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
