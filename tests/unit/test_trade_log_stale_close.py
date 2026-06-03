from __future__ import annotations

from pathlib import Path

from rlm.execution.trade_log_io import close_stale_open_rows_above_dte, open_plan_ids


def test_close_stale_open_rows_above_dte(tmp_path: Path) -> None:
    log = tmp_path / "log.csv"
    log.write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,peak_mark,"
        "unrealized_pnl,unrealized_pnl_pct,signal,closed,dte,legs_json\n"
        "2026-01-01T00:00:00Z,OLD_SPY,SPY,,100,100,100,100,0,0,hold,0,23.0,[]\n",
        encoding="utf-8",
    )
    n = close_stale_open_rows_above_dte(log, max_dte=5.0)
    assert n == 1
    assert open_plan_ids(log) == set()
