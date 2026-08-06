from rlm.execution.risk_targets import (
    build_spread_exit_thresholds,
    should_trailing_stop_exit,
    trailing_stop_from_peak,
)


def test_build_spread_exit_thresholds() -> None:
    t = build_spread_exit_thresholds(
        v0=100.0,
        entry_debit=200.0,
        target_profit_pct=0.5,
        stop_loss_frac_of_debit=0.5,
        trail_activate_frac_of_debit=0.30,
        trail_retrace_frac_from_peak=0.20,
        min_trail_profit_frac_of_debit=0.08,
    )
    assert t.v_take_profit == 100.0 + 0.5 * 200.0
    assert t.v_hard_stop == 100.0 - 0.5 * 200.0
    assert t.v_trail_activate == 100.0 + 0.30 * 200.0
    assert t.trail_retrace_frac == 0.20
    assert t.min_trail_exit_v == 100.0 + 0.08 * 200.0


def test_trailing_stop_from_peak() -> None:
    assert trailing_stop_from_peak(100.0, 0.2) == 80.0
    # Credit-style peak: stop must sit more negative than the peak (adverse side).
    assert trailing_stop_from_peak(-70.0, 0.2) == -84.0


def test_should_trailing_stop_exit_respects_profit_floor() -> None:
    # Peak 150, 20% retrace → stop 120; min exit 116
    assert not should_trailing_stop_exit(
        v=90.0,
        peak_v=150.0,
        retrace_frac=0.20,
        min_exit_v=116.0,
    )
    assert should_trailing_stop_exit(
        v=118.0,
        peak_v=150.0,
        retrace_frac=0.20,
        min_exit_v=116.0,
    )


def test_credit_trail_does_not_exit_on_arm_tick() -> None:
    """Credit V0=-100, |D|=100 → activate at -70; arming must not same-tick exit."""
    t = build_spread_exit_thresholds(
        v0=-100.0,
        entry_debit=100.0,
        target_profit_pct=0.5,
        trail_activate_frac_of_debit=0.30,
        trail_retrace_frac_from_peak=0.20,
        min_trail_profit_frac_of_debit=0.08,
    )
    assert t.v_trail_activate == -70.0
    assert t.min_trail_exit_v == -92.0
    # Same tick the trail arms at the activate mark.
    assert not should_trailing_stop_exit(
        v=t.v_trail_activate,
        peak_v=t.v_trail_activate,
        retrace_frac=t.trail_retrace_frac,
        min_exit_v=t.min_trail_exit_v,
    )
    # Adverse retrace through the credit trail stop while still above the floor.
    assert should_trailing_stop_exit(
        v=-85.0,
        peak_v=t.v_trail_activate,
        retrace_frac=t.trail_retrace_frac,
        min_exit_v=t.min_trail_exit_v,
    )
    # Further adverse beyond the profit floor → leave to hard stop, not trail.
    assert not should_trailing_stop_exit(
        v=-95.0,
        peak_v=t.v_trail_activate,
        retrace_frac=t.trail_retrace_frac,
        min_exit_v=t.min_trail_exit_v,
    )
