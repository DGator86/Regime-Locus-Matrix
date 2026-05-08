"""Equity monitor: universe-absence grace and stop/target precedence."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _equity_script_module():
    p = ROOT / "scripts" / "ibkr_equity_paper_trade.py"
    name = "_ibkr_equity_paper_trade_testmod"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mk_open_pos(mod, *, plan_id: str = "plan_x"):
    base = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
    return base, mod.EquityPosition(
        plan_id=plan_id,
        symbol="QQQ",
        direction="bull",
        side="long",
        quantity=10,
        entry_price=400.0,
        entry_ts=base.isoformat(),
    )


def test_plan_absent_grace_waits_before_close(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
    )
    assert pos.status == "open"
    assert pos.plan_missing_since_utc is not None

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=599),
    )
    assert pos.status == "open"

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=600),
    )
    assert pos.status == "closed"
    assert pos.exit_reason == "plan_no_longer_active"


def test_plan_absent_zero_grace_closes_immediately(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=0.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
    )
    assert pos.status == "closed"


def test_stop_loss_before_universe_even_when_plan_missing(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    # Force loss beyond 5%
    pos.entry_price = 100.0
    positions = {pos.plan_id: pos}

    class FakeApp:
        def get_last_price(self, symbol: str) -> float:
            assert symbol == "QQQ"
            return 94.0

    fake = FakeApp()

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=99999.0,
        dry_run=True,
        app=fake,
        log_path=log_path,
        utc_now=t0,
    )
    assert pos.exit_reason == "stop_loss_5.0pct"
    assert pos.status == "closed"


def test_plan_returns_clears_grace_timer(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
    )
    assert pos.plan_missing_since_utc is not None

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=10000),
    )
    assert pos.plan_missing_since_utc is None
    assert pos.status == "open"
