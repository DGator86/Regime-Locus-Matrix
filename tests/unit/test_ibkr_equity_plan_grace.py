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


def test_mark_equity_opened_does_not_overwrite_fresher_universe_plans(tmp_path: Path) -> None:
    """Regression: equity fill must not RMW-stamp universe_trade_plans.json.

    Concrete race (master --with-equity): startup equity thread still placing an
    IBKR order after the 120s join timeout while a universe rescan republishes
    plans. The old read-modify-write stamp could write a stale snapshot over the
    fresh scan and wipe active_ranked for the options monitor / Telegram.
    """
    import json
    import time

    mod = _equity_script_module()
    plans_path = tmp_path / "universe_trade_plans.json"
    stale = {
        "generated_at_utc": "2026-07-27T14:00:00+00:00",
        "active_ranked": [
            {"plan_id": "SPY_20260727_1400", "symbol": "SPY", "status": "active"},
            {"plan_id": "IWM_20260727_1400", "symbol": "IWM", "status": "active"},
        ],
        "results": [
            {"plan_id": "SPY_20260727_1400", "symbol": "SPY", "status": "active"},
            {"plan_id": "IWM_20260727_1400", "symbol": "IWM", "status": "active"},
        ],
    }
    fresh = {
        "generated_at_utc": "2026-07-27T14:05:00+00:00",
        "active_ranked": [
            {"plan_id": "QQQ_20260727_1405", "symbol": "QQQ", "status": "active"},
        ],
        "results": [
            {"plan_id": "QQQ_20260727_1405", "symbol": "QQQ", "status": "active"},
            {"plan_id": "IWM_20260727_1405", "symbol": "IWM", "status": "skipped"},
        ],
    }
    plans_path.write_text(json.dumps(stale, indent=2), encoding="utf-8")
    # Old bug: read stale snapshot, then a concurrent rescan publishes fresh…
    held = json.loads(plans_path.read_text(encoding="utf-8"))
    for section in ("active_ranked", "results"):
        for row in held.get(section) or []:
            if row.get("plan_id") == "SPY_20260727_1400":
                row["equity_opened"] = True
    plans_path.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
    # …and writing the held snapshot would wipe the fresh scan (document hazard).
    wiped = json.dumps(held, indent=2)
    assert "QQQ_20260727_1405" not in wiped

    before = plans_path.read_text(encoding="utf-8")
    time.sleep(0.01)
    # Stamp target still present in an alternate concurrent case: must not mutate.
    mod._mark_equity_opened("QQQ_20260727_1405", plans_path)
    after = plans_path.read_text(encoding="utf-8")
    assert after == before
    payload = json.loads(after)
    assert payload["generated_at_utc"] == fresh["generated_at_utc"]
    assert [r["plan_id"] for r in payload["active_ranked"]] == ["QQQ_20260727_1405"]
    assert len(payload["results"]) == 2
    assert "equity_opened" not in json.dumps(payload)

def test_open_equity_skips_plan_id_already_open_in_state(tmp_path: Path) -> None:
    mod = _equity_script_module()
    log_path = tmp_path / "equity_trade_log.csv"
    plans_path = tmp_path / "universe_trade_plans.json"
    plans_path.write_text("{}", encoding="utf-8")
    _, pos = _mk_open_pos(mod, plan_id="SPY_KEEP")
    positions = {pos.plan_id: pos}
    plans = [
        {
            "plan_id": "SPY_KEEP",
            "symbol": "SPY",
            "status": "active",
            "regime_direction": "bull",
            "regime_key": "bull|trend",
            "last_close": 500.0,
        }
    ]
    before_len = len(positions)
    mod.open_equity_positions(
        plans=plans,
        positions=positions,
        position_usd=1000.0,
        dry_run=True,
        app=None,
        plans_path=plans_path,
        log_path=log_path,
    )
    assert len(positions) == before_len
    assert positions["SPY_KEEP"].quantity == pos.quantity
