from __future__ import annotations

import pytest

from rlm.roee.exits import (
    should_exit_for_profit,
    should_exit_for_regime_flip,
    should_exit_for_stop_loss,
    should_exit_for_time_stop,
    should_exit_for_zone_breach,
)


# ---------------------------------------------------------------------------
# should_exit_for_regime_flip
# ---------------------------------------------------------------------------


class TestRegimeFlip:
    def test_same_direction_no_exit(self) -> None:
        assert not should_exit_for_regime_flip(
            "bull|high_vol|low_liquidity|unknown",
            "bull|low_vol|high_liquidity|supportive",
        )

    def test_true_reversal_bull_to_bear(self) -> None:
        assert should_exit_for_regime_flip(
            "bull|high_vol|low_liquidity|unknown",
            "bear|high_vol|low_liquidity|unknown",
        )

    def test_true_reversal_bear_to_bull(self) -> None:
        assert should_exit_for_regime_flip(
            "bear|low_vol|high_liquidity|supportive",
            "bull|low_vol|high_liquidity|supportive",
        )

    def test_bull_to_range_exits(self) -> None:
        # Direction changed from committed bull to range — still an exit.
        assert should_exit_for_regime_flip(
            "bull|high_vol|low_liquidity|unknown",
            "range|high_vol|low_liquidity|unknown",
        )

    def test_range_to_bear_exits(self) -> None:
        assert should_exit_for_regime_flip(
            "range|low_vol|high_liquidity|supportive",
            "bear|low_vol|high_liquidity|supportive",
        )

    # --- transition hold tests (root-cause regression guard) ---

    def test_bull_to_transition_no_exit(self) -> None:
        # S_D drops from >0.6 to 0.3-0.6 zone — signal uncertainty, not reversal.
        assert not should_exit_for_regime_flip(
            "bull|high_vol|low_liquidity|unknown",
            "transition|high_vol|low_liquidity|unknown",
        )

    def test_bear_to_transition_no_exit(self) -> None:
        assert not should_exit_for_regime_flip(
            "bear|high_vol|low_liquidity|unknown",
            "transition|high_vol|low_liquidity|unknown",
        )

    def test_range_to_transition_no_exit(self) -> None:
        assert not should_exit_for_regime_flip(
            "range|low_vol|high_liquidity|supportive",
            "transition|low_vol|high_liquidity|supportive",
        )

    def test_transition_to_bull_no_exit(self) -> None:
        # Entry was in transition; resolving to a committed state is fine.
        assert not should_exit_for_regime_flip(
            "transition|high_vol|low_liquidity|unknown",
            "bull|high_vol|low_liquidity|unknown",
        )

    def test_transition_to_bear_no_exit(self) -> None:
        assert not should_exit_for_regime_flip(
            "transition|high_vol|low_liquidity|unknown",
            "bear|high_vol|low_liquidity|unknown",
        )

    def test_transition_to_transition_no_exit(self) -> None:
        assert not should_exit_for_regime_flip(
            "transition|high_vol|low_liquidity|unknown",
            "transition|low_vol|high_liquidity|supportive",
        )

    def test_plain_keys_same_no_exit(self) -> None:
        assert not should_exit_for_regime_flip("bull", "bull")

    def test_plain_keys_different_exits(self) -> None:
        assert should_exit_for_regime_flip("bull", "bear")


# ---------------------------------------------------------------------------
# should_exit_for_profit
# ---------------------------------------------------------------------------


class TestProfitExit:
    def test_at_target(self) -> None:
        assert should_exit_for_profit(0.45, 0.45)

    def test_above_target(self) -> None:
        assert should_exit_for_profit(0.60, 0.45)

    def test_below_target(self) -> None:
        assert not should_exit_for_profit(0.30, 0.45)


# ---------------------------------------------------------------------------
# should_exit_for_stop_loss
# ---------------------------------------------------------------------------


class TestStopLossExit:
    def test_at_stop(self) -> None:
        assert should_exit_for_stop_loss(-0.50, -0.50)

    def test_beyond_stop(self) -> None:
        assert should_exit_for_stop_loss(-0.55, -0.50)

    def test_above_stop(self) -> None:
        assert not should_exit_for_stop_loss(-0.30, -0.50)


# ---------------------------------------------------------------------------
# should_exit_for_time_stop
# ---------------------------------------------------------------------------


class TestTimeStopExit:
    def test_at_threshold(self) -> None:
        assert should_exit_for_time_stop(2.0, 2.0)

    def test_below_threshold(self) -> None:
        assert should_exit_for_time_stop(1.0, 2.0)

    def test_above_threshold(self) -> None:
        assert not should_exit_for_time_stop(5.0, 2.0)


# ---------------------------------------------------------------------------
# should_exit_for_zone_breach
# ---------------------------------------------------------------------------


class TestZoneBreachExit:
    def test_below_lower(self) -> None:
        assert should_exit_for_zone_breach(490.0, 495.0, 510.0)

    def test_above_upper(self) -> None:
        assert should_exit_for_zone_breach(515.0, 495.0, 510.0)

    def test_within_zone(self) -> None:
        assert not should_exit_for_zone_breach(502.0, 495.0, 510.0)
