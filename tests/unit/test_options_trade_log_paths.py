from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlm.notify.options_paths import options_trade_log_read_paths  # noqa: E402
from rlm.notify.pnl_report import calculate_daily_pnl  # noqa: E402


def test_read_paths_orders_primary_then_fallback(tmp_path: Path) -> None:
    (tmp_path / "data" / "processed").mkdir(parents=True)
    primary = tmp_path / "data" / "processed" / "trade_log.csv"
    fb = tmp_path / "data" / "processed" / "options_large_account_trade_log.csv"
    primary.write_text("timestamp_utc,plan_id\n", encoding="utf-8")
    cands = options_trade_log_read_paths(tmp_path)
    assert cands[0] == primary
    assert cands[-1] == fb


def test_eod_reads_fallback_when_primary_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dproc = tmp_path / "data" / "processed"
    dproc.mkdir(parents=True)
    (dproc / "trade_log.csv").write_text("timestamp_utc,plan_id,symbol\n", encoding="utf-8")
    h = (
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n"
    )
    rows = [
        "2026-04-24T18:00:00Z,p1,SPY,x,1,1,0.9,0.9,-0.1,-1,hold,0,1\n",
    ]
    (dproc / "options_large_account_trade_log.csv").write_text(h + "".join(rows), encoding="utf-8")
    fixed = datetime(2026, 4, 24, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("rlm.notify.pnl_report._now_utc", lambda: fixed, raising=True)
    text = calculate_daily_pnl(tmp_path)
    assert "SPY" in text
    assert "source:" in text.lower()
