"""Equity monitor: universe-absence grace, regime/transition, stop/target precedence."""

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
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )
    assert pos.status == "open"
    assert pos.plan_missing_since_utc is not None

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=599),
        exit_on_plan_absent=True,
    )
    assert pos.status == "open"

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=600),
        exit_on_plan_absent=True,
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
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=0.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
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
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=fake,
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )
    assert pos.exit_reason is not None
    assert str(pos.exit_reason).startswith("stop_loss_") and pos.exit_reason.endswith("pct")
    assert pos.status == "closed"


def test_plan_returns_clears_grace_timer(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )
    assert pos.plan_missing_since_utc is not None

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=10000),
        exit_on_plan_absent=True,
    )
    assert pos.plan_missing_since_utc is None
    assert pos.status == "open"


def test_regime_flip_exits_while_plan_still_active(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_regime_key = "bull|tu|rv|dl"
    positions = {pos.plan_id: pos}
    plan_row = {
        "plan_id": pos.plan_id,
        "regime_key": "bear|tu|rv|dl",
        "pipeline": {
            "regime_transition": {"family": "hmm", "most_likely_next_prob": 0.5},
        },
    }
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={pos.plan_id: plan_row},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        min_hold_sec=0.0,
    )
    assert pos.status == "closed"
    assert pos.exit_reason == "regime_flip"


def test_weak_transition_top1_prob_exit(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_regime_key = "bull|tu|rv|dl"
    positions = {pos.plan_id: pos}
    plan_row = {
        "plan_id": pos.plan_id,
        "regime_key": "bull|tu|rv|dl",
        "pipeline": {
            "regime_transition": {"family": "hmm", "most_likely_next_prob": 0.05},
        },
    }
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={pos.plan_id: plan_row},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=0.1,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        min_hold_sec=0.0,
    )
    assert pos.status == "closed"
    assert pos.exit_reason == "weak_transition_top1_prob"


def test_weak_transition_label_mass_exit(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_regime_key = "bull|tu|rv|dl"
    positions = {pos.plan_id: pos}
    plan_row = {
        "plan_id": pos.plan_id,
        "regime_key": "bull|tu|rv|dl",
        "pipeline": {
            "regime_transition": {
                "family": "hmm",
                "most_likely_next_prob": 0.9,
                "next_label_aligned_bull_mass": 0.12,
                "next_label_aligned_bear_mass": 0.88,
            },
        },
    }
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={pos.plan_id: plan_row},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=0.2,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0,
        min_hold_sec=0.0,
    )
    assert pos.status == "closed"
    assert pos.exit_reason == "weak_transition_label_mass"


def test_plan_absent_without_exit_flag_never_closes(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids=set(),
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=600.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(days=3),
        exit_on_plan_absent=False,
    )
    assert pos.status == "open"
    assert pos.plan_missing_since_utc is None


def test_min_hold_defers_regime_flip(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_regime_key = "bull|tu|rv|dl"
    positions = {pos.plan_id: pos}
    plan_row = {
        "plan_id": pos.plan_id,
        "regime_key": "bear|tu|rv|dl",
        "pipeline": {},
    }
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={pos.plan_id: plan_row},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=30),
        min_hold_sec=120.0,
    )
    assert pos.status == "open"

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={pos.plan_id: plan_row},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=None,
        log_path=log_path,
        utc_now=t0 + timedelta(seconds=200),
        min_hold_sec=120.0,
    )
    assert pos.status == "closed"
    assert pos.exit_reason == "regime_flip"


def test_trailing_giveback_long(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    positions = {pos.plan_id: pos}

    class FakeApp:
        price = 420.0

        def get_last_price(self, symbol: str) -> float:
            return self.price

    fake = FakeApp()
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=fake,
        log_path=log_path,
        utc_now=t0,
        trail_activate_pct=4.0,
        trail_retrace_frac=0.35,
    )
    assert pos.status == "open"
    assert pos.trail_armed is True

    fake.price = 411.0
    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={},
        stop_pct=50.0,
        target_pct=50.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=True,
        app=fake,
        log_path=log_path,
        utc_now=t0 + timedelta(minutes=1),
        trail_activate_pct=4.0,
        trail_retrace_frac=0.35,
    )
    assert pos.status == "closed"
    assert pos.exit_reason is not None
    assert pos.exit_reason.startswith("trailing_giveback_")


def test_ibkr_order_outcome_requires_fill_not_submitted_or_cancelled() -> None:
    mod = _equity_script_module()
    assert mod._ibkr_order_outcome("PreSubmitted") == "pending"
    assert mod._ibkr_order_outcome("Submitted") == "pending"
    assert mod._ibkr_order_outcome("Cancelled") == "failed"
    assert mod._ibkr_order_outcome("ApiCancelled") == "failed"
    assert mod._ibkr_order_outcome("Rejected") == "failed"
    assert mod._ibkr_order_outcome("Filled", filled=10.0, remaining=0.0) == "filled"
    assert mod._ibkr_order_outcome("Submitted", filled=10.0, remaining=0.0) == "filled"


def test_cancelled_live_close_keeps_position_open_for_retry(tmp_path: Path) -> None:
    """Cancelled/unfilled close must not mark local state closed (lost exposure)."""
    import csv

    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_price = 100.0
    positions = {pos.plan_id: pos}

    class CancelCloseApp:
        def get_last_price(self, symbol: str) -> float:
            assert symbol == "QQQ"
            return 94.0

        def place_stock_order(self, symbol: str, action: str, quantity: int) -> int:
            assert (symbol, action, quantity) == ("QQQ", "SELL", 10)
            return 77

        def wait_for_order(self, order_id: int) -> dict:
            assert order_id == 77
            raise RuntimeError("IBKR order 77 Cancelled: Cancelled")

    mod.evaluate_equity_positions(
        positions=positions,
        active_plan_ids={pos.plan_id},
        plan_by_id={},
        stop_pct=5.0,
        target_pct=10.0,
        grace_sec=99999.0,
        min_most_likely_next_prob=None,
        min_next_label_aligned_mass=None,
        dry_run=False,
        app=CancelCloseApp(),
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )

    assert pos.status == "open"
    assert pos.exit_reason is None
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8", newline="")))
    assert rows[-1]["signal"] == "close_order_error"
    assert rows[-1]["closed"] == "0"
    assert "Cancelled" in rows[-1]["note"]


def test_unfilled_live_open_does_not_record_position(tmp_path: Path) -> None:
    """Submitted/Cancelled opens must not create ghost equity_positions_state rows."""
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    plans_path = tmp_path / "universe_trade_plans.json"
    plans_path.write_text("{}", encoding="utf-8")
    positions: dict = {}
    plan = {
        "plan_id": "NVDA_20260808_1000",
        "symbol": "NVDA",
        "regime_key": "bull|trend",
        "regime_direction": "bull",
        "pipeline": {"close": 100.0},
        "decision": {"metadata": {"regime_confidence": 0.9}},
    }

    class UnfilledOpenApp:
        def place_stock_order(
            self, symbol: str, action: str, quantity: int, limit_price: float | None = None
        ) -> int:
            return 88

        def wait_for_order(self, order_id: int) -> dict:
            raise RuntimeError("IBKR order 88 Cancelled: Cancelled")

    mod.open_equity_positions(
        plans=[plan],
        positions=positions,
        position_usd=1000.0,
        dry_run=False,
        app=UnfilledOpenApp(),
        plans_path=plans_path,
        log_path=log_path,
    )
    assert positions == {}
    assert not log_path.is_file()
