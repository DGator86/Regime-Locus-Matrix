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
    # Partial fill then cancel must book the filled shares (not "failed").
    assert mod._ibkr_order_outcome("Cancelled", filled=4.0, remaining=6.0) == "filled"
    assert mod._ibkr_order_outcome("ApiCancelled", filled=4.0, remaining=6.0) == "filled"


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


def test_fill_meta_dict_rejects_legacy_status_trail() -> None:
    """Legacy wait_for_order list returns cannot prove fill qty."""
    mod = _equity_script_module()
    meta = mod._fill_meta_dict(["PreSubmitted", "Submitted", "Filled"])
    assert meta["status"] == "Filled"
    assert meta["filled"] == 0.0
    assert meta["avg_fill_price"] == 0.0


def test_submitted_without_fill_qty_skips_open(tmp_path: Path) -> None:
    """wait_for_order dict with filled=0 (working Submitted) must not open locally."""
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    plans_path = tmp_path / "universe_trade_plans.json"
    plans_path.write_text("{}", encoding="utf-8")
    positions: dict = {}
    plan = {
        "plan_id": "NVDA_20260808_1001",
        "symbol": "NVDA",
        "regime_key": "bull|trend",
        "regime_direction": "bull",
        "pipeline": {"close": 100.0},
        "decision": {"metadata": {"regime_confidence": 0.9}},
    }

    class SubmittedOpenApp:
        def place_stock_order(
            self, symbol: str, action: str, quantity: int, limit_price: float | None = None
        ) -> int:
            return 89

        def wait_for_order(self, order_id: int) -> dict:
            return {
                "status": "Submitted",
                "filled": 0.0,
                "remaining": 10.0,
                "avg_fill_price": 0.0,
                "trail": ["PreSubmitted", "Submitted"],
            }

    mod.open_equity_positions(
        plans=[plan],
        positions=positions,
        position_usd=1000.0,
        dry_run=False,
        app=SubmittedOpenApp(),
        plans_path=plans_path,
        log_path=log_path,
    )
    assert positions == {}


def test_legacy_list_wait_for_order_does_not_crash_or_open(tmp_path: Path) -> None:
    """Regression: list trail + .get('filled') used to AttributeError on live opens."""
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    plans_path = tmp_path / "universe_trade_plans.json"
    plans_path.write_text("{}", encoding="utf-8")
    positions: dict = {}
    plan = {
        "plan_id": "NVDA_20260808_1002",
        "symbol": "NVDA",
        "regime_key": "bull|trend",
        "regime_direction": "bull",
        "pipeline": {"close": 100.0},
        "decision": {"metadata": {"regime_confidence": 0.9}},
    }

    class LegacyListOpenApp:
        def place_stock_order(
            self, symbol: str, action: str, quantity: int, limit_price: float | None = None
        ) -> int:
            return 90

        def wait_for_order(self, order_id: int) -> list:
            return ["PreSubmitted", "Submitted", "Filled"]

    mod.open_equity_positions(
        plans=[plan],
        positions=positions,
        position_usd=1000.0,
        dry_run=False,
        app=LegacyListOpenApp(),
        plans_path=plans_path,
        log_path=log_path,
    )
    assert positions == {}


def test_filled_live_open_records_avg_fill_price(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    plans_path = tmp_path / "universe_trade_plans.json"
    plans_path.write_text("{}", encoding="utf-8")
    positions: dict = {}
    plan = {
        "plan_id": "NVDA_20260808_1003",
        "symbol": "NVDA",
        "regime_key": "bull|trend",
        "regime_direction": "bull",
        "pipeline": {"close": 100.0},
        "decision": {"metadata": {"regime_confidence": 0.9}},
    }

    class FilledOpenApp:
        def place_stock_order(
            self, symbol: str, action: str, quantity: int, limit_price: float | None = None
        ) -> int:
            return 91

        def wait_for_order(self, order_id: int) -> dict:
            return {
                "status": "Filled",
                "filled": 10.0,
                "remaining": 0.0,
                "avg_fill_price": 101.25,
                "trail": ["Submitted", "Filled"],
            }

    mod.open_equity_positions(
        plans=[plan],
        positions=positions,
        position_usd=1000.0,
        dry_run=False,
        app=FilledOpenApp(),
        plans_path=plans_path,
        log_path=log_path,
    )
    assert "NVDA_20260808_1003" in positions
    pos = positions["NVDA_20260808_1003"]
    assert pos.status == "open"
    assert pos.quantity == 10
    assert pos.entry_price == 101.25


def test_filled_live_close_marks_position_closed(tmp_path: Path) -> None:
    import csv

    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_price = 100.0
    positions = {pos.plan_id: pos}

    class FilledCloseApp:
        def get_last_price(self, symbol: str) -> float:
            return 94.0

        def place_stock_order(self, symbol: str, action: str, quantity: int) -> int:
            return 92

        def wait_for_order(self, order_id: int) -> dict:
            return {
                "status": "Filled",
                "filled": 10.0,
                "remaining": 0.0,
                "avg_fill_price": 93.5,
                "trail": ["Submitted", "Filled"],
            }

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
        app=FilledCloseApp(),
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )

    assert pos.status == "closed"
    assert pos.exit_reason == "stop_loss_5pct"
    assert pos.exit_price == 93.5
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8", newline="")))
    assert rows[-1]["closed"] == "1"
    assert rows[-1]["signal"] == "closed"


def test_wait_for_order_timeout_cancels_working_order() -> None:
    """Timeout must cancel the live order so the next tick cannot double-buy/sell."""
    import pytest

    mod = _equity_script_module()

    class FakeInner:
        def __init__(self) -> None:
            self._order_status: dict[int, list[str]] = {}
            self._order_meta: dict[int, dict] = {}
            self._error_lines: list = []
            self.cancelled: list[int] = []

        def cancelOrder(self, order_id: int, *args: object) -> None:
            self.cancelled.append(int(order_id))
            self._order_meta[int(order_id)] = {
                "status": "Cancelled",
                "filled": 0.0,
                "remaining": 10.0,
                "avg_fill_price": 0.0,
            }
            self._order_status.setdefault(int(order_id), []).append("Cancelled")

    app = object.__new__(mod._EquityApp)
    app._app = FakeInner()
    app._app._order_meta[7] = {
        "status": "Submitted",
        "filled": 0.0,
        "remaining": 10.0,
        "avg_fill_price": 0.0,
    }
    app._app._order_status[7] = ["Submitted"]

    with pytest.raises(RuntimeError, match="timeout"):
        app.wait_for_order(7, timeout_sec=0.25)
    assert app._app.cancelled == [7]


def test_wait_for_order_timeout_returns_late_fill_after_cancel() -> None:
    """If the order fills while we cancel after timeout, book the fill."""
    mod = _equity_script_module()

    class FakeInner:
        def __init__(self) -> None:
            self._order_status: dict[int, list[str]] = {}
            self._order_meta: dict[int, dict] = {}
            self._error_lines: list = []
            self.cancelled: list[int] = []

        def cancelOrder(self, order_id: int, *args: object) -> None:
            self.cancelled.append(int(order_id))
            self._order_meta[int(order_id)] = {
                "status": "Filled",
                "filled": 10.0,
                "remaining": 0.0,
                "avg_fill_price": 101.5,
            }
            self._order_status.setdefault(int(order_id), []).append("Filled")

    app = object.__new__(mod._EquityApp)
    app._app = FakeInner()
    app._app._order_meta[8] = {
        "status": "Submitted",
        "filled": 0.0,
        "remaining": 10.0,
        "avg_fill_price": 0.0,
    }
    app._app._order_status[8] = ["Submitted"]

    fill = app.wait_for_order(8, timeout_sec=0.25)
    assert app._app.cancelled == [8]
    assert fill["filled"] == 10.0
    assert fill["avg_fill_price"] == 101.5


def test_partial_live_close_keeps_residual_open(tmp_path: Path) -> None:
    """Partial close fill must shrink qty and retry — not mark fully closed."""
    import csv

    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    t0, pos = _mk_open_pos(mod)
    pos.entry_price = 100.0
    pos.quantity = 10
    positions = {pos.plan_id: pos}

    class PartialCloseApp:
        def get_last_price(self, symbol: str) -> float:
            return 94.0

        def place_stock_order(self, symbol: str, action: str, quantity: int) -> int:
            assert quantity == 10
            return 93

        def wait_for_order(self, order_id: int) -> dict:
            return {
                "status": "Cancelled",
                "filled": 4.0,
                "remaining": 6.0,
                "avg_fill_price": 93.5,
                "trail": ["Submitted", "Cancelled"],
            }

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
        app=PartialCloseApp(),
        log_path=log_path,
        utc_now=t0,
        exit_on_plan_absent=True,
    )

    assert pos.status == "open"
    assert pos.quantity == 6
    rows = list(csv.DictReader(log_path.open("r", encoding="utf-8", newline="")))
    assert rows[-1]["signal"] == "partial_close"
    assert rows[-1]["closed"] == "0"
    assert rows[-1]["quantity"] == "4"
