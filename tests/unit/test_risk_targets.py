from rlm.execution.risk_targets import build_spread_exit_thresholds, trailing_stop_from_peak


def test_build_spread_exit_thresholds() -> None:
    t = build_spread_exit_thresholds(
        v0=100.0,
        entry_debit=200.0,
        target_profit_pct=0.5,
        stop_loss_frac_of_debit=0.5,
        trail_activate_frac_of_debit=0.15,
        trail_retrace_frac_from_peak=0.25,
    )
    assert t.v_take_profit == 100.0 + 0.5 * 200.0
    assert t.v_hard_stop == 100.0 - 0.5 * 200.0
    assert t.v_trail_activate == 100.0 + 0.15 * 200.0
    assert t.trail_retrace_frac == 0.25


def test_trailing_stop_from_peak() -> None:
    assert trailing_stop_from_peak(100.0, 0.2) == 80.0
