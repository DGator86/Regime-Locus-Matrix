from __future__ import annotations

import csv
from pathlib import Path

from scripts.check_performance_and_retune import _read_closed_pnl, _win_rate


def _write_log(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["timestamp_utc", "plan_id", "unrealized_pnl", "closed"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_read_closed_pnl_counts_each_plan_once(tmp_path: Path) -> None:
    log = tmp_path / "trade_log.csv"
    _write_log(
        log,
        [
            {"timestamp_utc": "2026-01-01T14:00:00Z", "plan_id": "A", "unrealized_pnl": -10, "closed": "0"},
            {"timestamp_utc": "2026-01-01T15:00:00Z", "plan_id": "A", "unrealized_pnl": -12, "closed": "1"},
            {"timestamp_utc": "2026-01-01T15:01:00Z", "plan_id": "A", "unrealized_pnl": -11, "closed": "1"},
            {"timestamp_utc": "2026-01-01T15:02:00Z", "plan_id": "A", "unrealized_pnl": -9, "closed": "1"},
            {"timestamp_utc": "2026-01-01T16:00:00Z", "plan_id": "B", "unrealized_pnl": 5, "closed": "1"},
        ],
    )

    pnls = _read_closed_pnl(log, lookback=20)

    assert pnls == [-9.0, 5.0]
    assert _win_rate(pnls) == 0.5


def test_read_closed_pnl_uses_last_distinct_closed_trades(tmp_path: Path) -> None:
    log = tmp_path / "trade_log.csv"
    _write_log(
        log,
        [
            {"timestamp_utc": "2026-01-01T15:00:00Z", "plan_id": "A", "unrealized_pnl": 10, "closed": "1"},
            {"timestamp_utc": "2026-01-01T15:01:00Z", "plan_id": "B", "unrealized_pnl": -8, "closed": "1"},
            {"timestamp_utc": "2026-01-01T15:02:00Z", "plan_id": "C", "unrealized_pnl": 7, "closed": "1"},
            {"timestamp_utc": "2026-01-01T15:03:00Z", "plan_id": "B", "unrealized_pnl": -6, "closed": "1"},
        ],
    )

    assert _read_closed_pnl(log, lookback=2) == [7.0, -6.0]
