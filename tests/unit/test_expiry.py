"""Tests for the expiry settlement module."""
from __future__ import annotations

import pytest

from rlm.backtest.expiry import (
    SettlementResult,
    leg_intrinsic_value,
    settle_butterfly,
    settle_iron_condor,
    settle_legs_at_expiry,
    settle_strangle,
    settle_vertical_spread,
)

# ---------------------------------------------------------------------------
# leg_intrinsic_value
# ---------------------------------------------------------------------------


class TestLegIntrinsicValue:
    def test_long_call_itm(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="call", strike=100.0, underlying_price=110.0) == pytest.approx(10.0)

    def test_long_call_otm(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="call", strike=100.0, underlying_price=90.0) == 0.0

    def test_short_call_itm(self) -> None:
        # Short ITM call: negative contribution (portfolio owes)
        assert leg_intrinsic_value(side="short", option_type="call", strike=100.0, underlying_price=115.0) == pytest.approx(-15.0)

    def test_short_call_otm(self) -> None:
        assert leg_intrinsic_value(side="short", option_type="call", strike=100.0, underlying_price=95.0) == 0.0

    def test_long_put_itm(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="put", strike=100.0, underlying_price=85.0) == pytest.approx(15.0)

    def test_long_put_otm(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="put", strike=100.0, underlying_price=105.0) == 0.0

    def test_short_put_itm(self) -> None:
        assert leg_intrinsic_value(side="short", option_type="put", strike=100.0, underlying_price=90.0) == pytest.approx(-10.0)

    def test_short_put_otm(self) -> None:
        assert leg_intrinsic_value(side="short", option_type="put", strike=100.0, underlying_price=110.0) == 0.0

    def test_at_the_money_call(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="call", strike=100.0, underlying_price=100.0) == 0.0

    def test_at_the_money_put(self) -> None:
        assert leg_intrinsic_value(side="long", option_type="put", strike=100.0, underlying_price=100.0) == 0.0


# ---------------------------------------------------------------------------
# settle_legs_at_expiry — basic cases
# ---------------------------------------------------------------------------


class TestSettleLegsAtExpiry:
    def test_single_long_call_itm(self) -> None:
        legs = [{"side": "long", "option_type": "call", "strike": 100.0}]
        result = settle_legs_at_expiry(legs=legs, underlying_price=110.0, contract_multiplier=100)
        assert isinstance(result, SettlementResult)
        assert result.intrinsic_value == pytest.approx(10.0)
        assert result.cash_impact == pytest.approx(1000.0)
        assert not result.assignment_occurred

    def test_single_long_call_otm_expires_worthless(self) -> None:
        legs = [{"side": "long", "option_type": "call", "strike": 100.0}]
        result = settle_legs_at_expiry(legs=legs, underlying_price=90.0, contract_multiplier=100)
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0
        assert not result.assignment_occurred
        assert "worthless" in result.notes

    def test_single_short_call_itm_triggers_assignment(self) -> None:
        legs = [{"side": "short", "option_type": "call", "strike": 100.0}]
        result = settle_legs_at_expiry(legs=legs, underlying_price=105.0, contract_multiplier=100)
        assert result.intrinsic_value == pytest.approx(-5.0)
        assert result.cash_impact == pytest.approx(-500.0)
        assert result.assignment_occurred
        assert "assigned" in result.notes

    def test_single_short_call_otm_full_profit(self) -> None:
        legs = [{"side": "short", "option_type": "call", "strike": 100.0}]
        result = settle_legs_at_expiry(legs=legs, underlying_price=95.0, contract_multiplier=100)
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0
        assert not result.assignment_occurred

    def test_single_long_put_itm(self) -> None:
        legs = [{"side": "long", "option_type": "put", "strike": 100.0}]
        result = settle_legs_at_expiry(legs=legs, underlying_price=85.0, contract_multiplier=100)
        assert result.intrinsic_value == pytest.approx(15.0)
        assert result.cash_impact == pytest.approx(1500.0)

    def test_empty_legs_returns_zero(self) -> None:
        result = settle_legs_at_expiry(legs=[], underlying_price=100.0, contract_multiplier=100)
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0
        assert not result.assignment_occurred


# ---------------------------------------------------------------------------
# settle_vertical_spread
# ---------------------------------------------------------------------------


class TestSettleVerticalSpread:
    def test_bull_call_spread_max_profit(self) -> None:
        # Long 95 call, short 100 call; underlying settles at 105 (above both)
        result = settle_vertical_spread(
            long_strike=95.0,
            short_strike=100.0,
            option_type="call",
            underlying_price=105.0,
            contract_multiplier=100,
        )
        # Long 95 call intrinsic = 10; short 100 call intrinsic = -5
        # Net = +5 per share → +500 cash
        assert result.intrinsic_value == pytest.approx(5.0)
        assert result.cash_impact == pytest.approx(500.0)

    def test_bull_call_spread_max_loss(self) -> None:
        # Both legs OTM; spread expires worthless
        result = settle_vertical_spread(
            long_strike=95.0,
            short_strike=100.0,
            option_type="call",
            underlying_price=90.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0

    def test_bull_call_spread_partial_profit(self) -> None:
        # Underlying at 97: long 95 call ITM (+2), short 100 call OTM (0)
        result = settle_vertical_spread(
            long_strike=95.0,
            short_strike=100.0,
            option_type="call",
            underlying_price=97.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == pytest.approx(2.0)
        assert result.cash_impact == pytest.approx(200.0)

    def test_bear_put_spread_max_profit(self) -> None:
        # Long 110 put, short 100 put; underlying settles at 90 (below both)
        result = settle_vertical_spread(
            long_strike=110.0,
            short_strike=100.0,
            option_type="put",
            underlying_price=90.0,
            contract_multiplier=100,
        )
        # Long 110 put intrinsic = 20; short 100 put intrinsic = -10
        # Net = +10
        assert result.intrinsic_value == pytest.approx(10.0)
        assert result.cash_impact == pytest.approx(1000.0)

    def test_credit_call_spread_short_itm_assignment(self) -> None:
        # Short 95 call (sold), long 100 call (bought) — credit call spread
        # Underlying at 98: short 95 call ITM → assignment
        result = settle_vertical_spread(
            long_strike=100.0,
            short_strike=95.0,
            option_type="call",
            underlying_price=98.0,
            contract_multiplier=100,
        )
        # short 95 call intrinsic = -3; long 100 call intrinsic = 0
        assert result.intrinsic_value == pytest.approx(-3.0)
        assert result.assignment_occurred


# ---------------------------------------------------------------------------
# settle_iron_condor
# ---------------------------------------------------------------------------


class TestSettleIronCondor:
    def test_iron_condor_all_otm_full_premium(self) -> None:
        # Underlying stays between short strikes: all legs expire worthless
        result = settle_iron_condor(
            long_put_strike=80.0,
            short_put_strike=90.0,
            short_call_strike=110.0,
            long_call_strike=120.0,
            underlying_price=100.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0
        assert not result.assignment_occurred

    def test_iron_condor_call_side_breached(self) -> None:
        # Underlying at 115: short call (110) is ITM, long call (120) is OTM
        result = settle_iron_condor(
            long_put_strike=80.0,
            short_put_strike=90.0,
            short_call_strike=110.0,
            long_call_strike=120.0,
            underlying_price=115.0,
            contract_multiplier=100,
        )
        # short call 110: intrinsic = -5; long call 120: 0
        # All put legs OTM
        assert result.intrinsic_value == pytest.approx(-5.0)
        assert result.cash_impact == pytest.approx(-500.0)
        assert result.assignment_occurred

    def test_iron_condor_put_side_breached(self) -> None:
        # Underlying at 85: short put (90) is ITM, long put (80) is OTM
        result = settle_iron_condor(
            long_put_strike=80.0,
            short_put_strike=90.0,
            short_call_strike=110.0,
            long_call_strike=120.0,
            underlying_price=85.0,
            contract_multiplier=100,
        )
        # short put 90: intrinsic = -5; long put 80: 0
        assert result.intrinsic_value == pytest.approx(-5.0)
        assert result.cash_impact == pytest.approx(-500.0)
        assert result.assignment_occurred

    def test_iron_condor_max_loss_call_side(self) -> None:
        # Underlying above long call: both call legs ITM
        result = settle_iron_condor(
            long_put_strike=80.0,
            short_put_strike=90.0,
            short_call_strike=110.0,
            long_call_strike=120.0,
            underlying_price=125.0,
            contract_multiplier=100,
        )
        # short call 110: intrinsic = -15; long call 120: intrinsic = +5
        # net call contribution = -10
        assert result.intrinsic_value == pytest.approx(-10.0)
        assert result.cash_impact == pytest.approx(-1000.0)


# ---------------------------------------------------------------------------
# settle_butterfly
# ---------------------------------------------------------------------------


class TestSettleButterfly:
    def test_long_call_butterfly_max_profit_at_middle(self) -> None:
        # Long 90 call, short 2x 100 call, long 110 call
        # Underlying at 100 (middle strike)
        result = settle_butterfly(
            lower_strike=90.0,
            middle_strike=100.0,
            upper_strike=110.0,
            option_type="call",
            underlying_price=100.0,
            contract_multiplier=100,
        )
        # long 90c: +10; short 100c x2: 0; long 110c: 0 → net = +10
        assert result.intrinsic_value == pytest.approx(10.0)
        assert result.cash_impact == pytest.approx(1000.0)

    def test_long_call_butterfly_max_loss_below_lower(self) -> None:
        # All legs OTM, expires worthless
        result = settle_butterfly(
            lower_strike=90.0,
            middle_strike=100.0,
            upper_strike=110.0,
            option_type="call",
            underlying_price=85.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == 0.0

    def test_long_call_butterfly_max_loss_above_upper(self) -> None:
        # Underlying above all strikes
        result = settle_butterfly(
            lower_strike=90.0,
            middle_strike=100.0,
            upper_strike=110.0,
            option_type="call",
            underlying_price=120.0,
            contract_multiplier=100,
        )
        # long 90c: +30; short 100c x2: -20 each = -40; long 110c: +10
        # net = 30 - 40 + 10 = 0
        assert result.intrinsic_value == pytest.approx(0.0)

    def test_long_put_butterfly(self) -> None:
        # Long 110 put, short 2x 100 put, long 90 put; underlying at 100
        result = settle_butterfly(
            lower_strike=90.0,
            middle_strike=100.0,
            upper_strike=110.0,
            option_type="put",
            underlying_price=100.0,
            contract_multiplier=100,
        )
        # long 90p: 0; short 100p x2: 0; long 110p: +10 → net = +10
        assert result.intrinsic_value == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# settle_strangle
# ---------------------------------------------------------------------------


class TestSettleStrangle:
    def test_long_strangle_both_otm_worthless(self) -> None:
        result = settle_strangle(
            put_strike=90.0,
            call_strike=110.0,
            side="long",
            underlying_price=100.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == 0.0
        assert result.cash_impact == 0.0

    def test_long_strangle_call_side_itm(self) -> None:
        result = settle_strangle(
            put_strike=90.0,
            call_strike=110.0,
            side="long",
            underlying_price=120.0,
            contract_multiplier=100,
        )
        # long call 110: +10; long put 90: 0
        assert result.intrinsic_value == pytest.approx(10.0)
        assert result.cash_impact == pytest.approx(1000.0)

    def test_long_strangle_put_side_itm(self) -> None:
        result = settle_strangle(
            put_strike=90.0,
            call_strike=110.0,
            side="long",
            underlying_price=80.0,
            contract_multiplier=100,
        )
        # long put 90: +10; long call 110: 0
        assert result.intrinsic_value == pytest.approx(10.0)

    def test_short_strangle_both_otm_full_profit(self) -> None:
        result = settle_strangle(
            put_strike=90.0,
            call_strike=110.0,
            side="short",
            underlying_price=100.0,
            contract_multiplier=100,
        )
        assert result.intrinsic_value == 0.0
        assert not result.assignment_occurred

    def test_short_strangle_call_side_assignment(self) -> None:
        result = settle_strangle(
            put_strike=90.0,
            call_strike=110.0,
            side="short",
            underlying_price=115.0,
            contract_multiplier=100,
        )
        # short call 110: -5
        assert result.intrinsic_value == pytest.approx(-5.0)
        assert result.cash_impact == pytest.approx(-500.0)
        assert result.assignment_occurred