"""Dynamic horizon blending for scripts/ibkr_equity_paper_trade.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _equity_script_module():
    p = ROOT / "scripts" / "ibkr_equity_paper_trade.py"
    name = "_ibkr_equity_dynamic_testmod"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bar_size_to_minutes() -> None:
    mod = _equity_script_module()
    assert mod._bar_size_to_minutes("5 mins") == 5.0
    assert mod._bar_size_to_minutes("1 hour") == 60.0
    assert mod._bar_size_to_minutes("1 day") == 1440.0
    assert mod._bar_size_to_minutes("") is None


def test_dynamic_plan_produces_wider_band_for_daily_than_5m() -> None:
    mod = _equity_script_module()
    fast_plan = {
        "pipeline": {
            "bar_size": "5 mins",
            "tf_confirmation_failed": True,
            "regime_transition": {"most_likely_next_prob": 0.25},
        },
        "decision": {"size_fraction": 0.55},
    }
    slow_plan = {
        "pipeline": {
            "bar_size": "1 day",
            "tf_confirmation_failed": False,
            "regime_transition": {"most_likely_next_prob": 0.9},
        },
        "decision": {"size_fraction": 0.95},
    }
    fs, ft, fm, _, _, _ = mod._resolve_equity_trade_params(
        fast_plan,
        dynamic_horizon=True,
        stop_pct=5.0,
        target_pct=10.0,
        min_hold_sec=1800.0,
        trail_activate_pct=4.0,
        trail_retrace_frac=0.35,
    )
    ss, st, sm, _, _, _ = mod._resolve_equity_trade_params(
        slow_plan,
        dynamic_horizon=True,
        stop_pct=5.0,
        target_pct=10.0,
        min_hold_sec=1800.0,
        trail_activate_pct=4.0,
        trail_retrace_frac=0.35,
    )
    assert fm < sm
    assert fs <= ss
    assert ft <= st


def test_resolve_static_matches_baseline_when_disabled() -> None:
    mod = _equity_script_module()
    plan = {"pipeline": {"bar_size": "1 day"}, "decision": {}}
    s, t, m, ta, tr, note = mod._resolve_equity_trade_params(
        plan,
        dynamic_horizon=False,
        stop_pct=3.5,
        target_pct=12.0,
        min_hold_sec=900.0,
        trail_activate_pct=4.0,
        trail_retrace_frac=0.35,
    )
    assert s == 3.5 and t == 12.0 and m == 900.0 and ta == 4.0 and tr == 0.35 and note == "static"
