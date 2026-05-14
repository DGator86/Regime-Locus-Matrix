"""Tests for ``rlm reset-paper``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def test_reset_paper_books_clears_json_and_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rlm.cli import reset_paper as rp

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(rp, "get_repo_root", lambda: repo)

    data = repo / "data"
    proc = data / "processed"
    proc.mkdir(parents=True)

    (proc / "equity_positions_state.json").write_text('{"x": 1}', encoding="utf-8")
    (proc / "trade_monitor_state.json").write_text('{"peak": 1}', encoding="utf-8")
    (proc / "telegram_notify_state.json").write_text('{"notify_seeded": true}', encoding="utf-8")
    (proc / "equity_trade_log.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    tlog = proc / "trade_log.csv"
    tlog.write_text(
        "timestamp_utc,plan_id,symbol,strategy,entry_debit,entry_mid,current_mark,"
        "peak_mark,unrealized_pnl,unrealized_pnl_pct,signal,closed,dte\n",
        encoding="utf-8",
    )

    ledgers = proc / "ledgers"
    ledgers.mkdir(parents=True)
    (ledgers / "large_options_book.csv").write_text("dummy\n", encoding="utf-8")

    rp.reset_paper_books(data_parent=data, archive=False, stamp="FIXTURE")

    assert json.loads((proc / "equity_positions_state.json").read_text(encoding="utf-8")) == {}
    assert json.loads((proc / "trade_monitor_state.json").read_text(encoding="utf-8")) == {}
    tg = json.loads((proc / "telegram_notify_state.json").read_text(encoding="utf-8"))
    assert tg.get("notify_seeded") is False
    assert tg.get("announced_trade_open") == []

    with (proc / "equity_trade_log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == []

    with tlog.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == []

    assert not (ledgers / "large_options_book.csv").exists()
