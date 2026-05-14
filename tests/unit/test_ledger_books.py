from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm.notify.ledger_books import (
    equity_book_snapshot,
    ledger_dir,
    options_book_snapshot,
    write_trading_ledgers,
)


def test_options_book_respects_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLM_LARGE_OPTIONS_BOOK_SEED", "250000")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    log = proc / "trade_log.csv"
    log.write_text(
        "plan_id,symbol,timestamp_utc,closed,unrealized_pnl,note\n" "p1,SPY,2026-01-02T15:00:00+00:00,0,-100,\n",
        encoding="utf-8",
    )
    snap = options_book_snapshot(tmp_path)
    assert snap.seed == 250_000.0
    assert snap.closed_realized == 0.0
    assert snap.open_mtm == -100.0
    assert snap.book_value == 249_900.0


def test_write_trading_ledgers_creates_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLM_LARGE_OPTIONS_BOOK_SEED", "100000")
    monkeypatch.setenv("RLM_LARGE_EQUITY_BOOK_SEED", "100000")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "trade_log.csv").write_text(
        "plan_id,symbol,timestamp_utc,closed,unrealized_pnl,note\n" "x,TSLA,2026-01-01T10:00:00+00:00,1,50,\n",
        encoding="utf-8",
    )
    (proc / "equity_trade_log.csv").write_text(
        "timestamp_utc,plan_id,symbol,action,quantity,current_mark,entry_debit,unrealized_pnl,signal,closed\n"
        "2026-01-01T11:00:00+00:00,e1,QQQ,BUY,10,100,100,5,hold,0\n",
        encoding="utf-8",
    )
    ch = tmp_path / "data" / "challenge"
    ch.mkdir(parents=True)
    (ch / "state.json").write_text(
        json.dumps(
            {
                "balance": 950.0,
                "seed": 1000.0,
                "target": 25000.0,
                "created_at": "2026-01-01T00:00:00Z",
                "trade_history": [],
                "open_positions": [],
            }
        ),
        encoding="utf-8",
    )
    paths = write_trading_ledgers(tmp_path)
    assert paths["large_options"].is_file()
    assert paths["large_equities"].is_file()
    assert paths["pdt_robinhood"].is_file()
    ld = ledger_dir(tmp_path)
    assert ld == tmp_path / "data" / "processed" / "ledgers"

    opt_snap = options_book_snapshot(tmp_path)
    eq_snap = equity_book_snapshot(tmp_path)
    assert opt_snap.book_value == pytest.approx(100_050.0 + 0.0)  # seed + closed + no open
    assert eq_snap.book_value == pytest.approx(100_000.0 + 0.0 + 5.0)


def test_equity_book_ignores_options_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLM_LARGE_EQUITY_BOOK_SEED", "250000")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "equity_trade_log.csv").write_text(
        "plan_id,symbol,timestamp_utc,action,quantity,unrealized_pnl,signal,closed\n"
        "o1,SPY,2026-01-02T15:00:00+00:00,BUY,1,1,open,0\n",  # missing strategy col like real log?
        encoding="utf-8",
    )
    monkeypatch.delenv("RLM_EQUITY_TRADE_LOG_PATH", raising=False)
    snap = equity_book_snapshot(tmp_path)
    assert snap.open_mtm == 1.0


def test_robinhood_elite_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlm.challenge.config import ChallengeConfig, apply_challenge_profile_env

    monkeypatch.setenv("RLM_CHALLENGE_PROFILE", "robinhood_elite")
    base = ChallengeConfig(seed_capital=1000.0, target_capital=25000.0)
    cfg = apply_challenge_profile_env(base)
    assert cfg.stage1_dte == 45
    assert cfg.stage1_size_frac == 0.12
    assert cfg.seed_capital == 1000.0
