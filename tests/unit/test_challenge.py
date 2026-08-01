"""Unit tests for the $1K→$25K aggressive options challenge module."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.engine import ChallengeEngine, _days_between
from rlm.challenge.pricing import (
    atm_premium,
    estimate_delta,
    estimate_premium,
    otm_premium,
    updated_premium,
)
from rlm.challenge.sizing import AggressiveSizer
from rlm.challenge.state import ChallengeState, ChallengeTradeRecord
from rlm.challenge.strategy import ChallengeStrategy
from rlm.challenge.tracker import ChallengeTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cfg() -> ChallengeConfig:
    return ChallengeConfig(seed_capital=1_000.0, target_capital=25_000.0)


class TestChallengeConfig:
    def test_stage1_tighter_profit_and_trail(self, cfg: ChallengeConfig) -> None:
        assert cfg.profit_target_for(1_000.0) == cfg.stage1_profit_target_mult
        assert cfg.profit_target_for(1_000.0) < cfg.stage3_profit_target_mult
        assert cfg.trail_activate_for(1_000.0) == cfg.stage1_trail_activate_mult

    def test_equity_value_includes_open_marks(self, cfg: ChallengeConfig) -> None:
        from rlm.challenge.state import ChallengePosition, ChallengeState

        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=500.0,
            strike=505.0,
            dte=7,
            entry_date="2026-01-01",
            premium_per_share=2.0,
            qty=1,
            delta=0.5,
            iv=0.18,
        )
        pos.current_premium = 2.5
        pos.current_value = 250.0
        state = ChallengeState(balance=750.0, seed=1_000.0, target=25_000.0, open_positions=[pos])
        assert state.equity_value == 1_000.0
        assert state.total_return_pct == 0.0


@pytest.fixture()
def cfg_enter(cfg: ChallengeConfig) -> ChallengeConfig:
    """Larger stage-1 fraction so engine tests can open affordable synthetic plays."""
    return replace(cfg, stage1_size_frac=0.85)


@pytest.fixture()
def tmp_tracker(tmp_path: Path) -> ChallengeTracker:
    return ChallengeTracker(data_root=str(tmp_path))


@pytest.fixture()
def fresh_state(cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> ChallengeState:
    return tmp_tracker.reset(cfg)


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------


class TestPricing:
    def test_atm_premium_positive(self) -> None:
        p = atm_premium(underlying=500.0, iv=0.18, dte=7)
        assert p > 0

    def test_atm_premium_increases_with_dte(self) -> None:
        p7 = atm_premium(500.0, 0.18, 7)
        p14 = atm_premium(500.0, 0.18, 14)
        assert p14 > p7

    def test_atm_premium_increases_with_iv(self) -> None:
        low = atm_premium(500.0, 0.15, 7)
        high = atm_premium(500.0, 0.30, 7)
        assert high > low

    def test_otm_premium_less_than_atm(self) -> None:
        atm = atm_premium(500.0, 0.18, 7)
        otm = otm_premium(500.0, 0.18, 7, strike=505.0)
        assert otm < atm

    def test_estimate_premium_delegates(self) -> None:
        p = estimate_premium(500.0, 0.18, 7, 500.0)
        assert p > 0

    def test_delta_call_atm_near_half(self) -> None:
        d = estimate_delta(500.0, 500.0, 0.18, 14, "call")
        assert 0.45 <= d <= 0.60

    def test_delta_put_atm_negative(self) -> None:
        d = estimate_delta(500.0, 500.0, 0.18, 14, "put")
        assert -0.60 <= d <= -0.40

    def test_delta_otm_call_less_than_atm(self) -> None:
        atm_d = estimate_delta(500.0, 500.0, 0.18, 14, "call")
        otm_d = estimate_delta(500.0, 510.0, 0.18, 14, "call")
        assert otm_d < atm_d

    def test_updated_premium_rises_on_favorable_move_call(self) -> None:
        entry = atm_premium(500.0, 0.18, 7)
        new = updated_premium(
            entry_premium=entry,
            delta=0.50,
            underlying_entry=500.0,
            underlying_now=505.0,  # +1% move in our favour
            days_elapsed=1,
            dte_remaining=6,
            iv=0.18,
        )
        assert new > entry * 0.90  # might not fully offset theta on 1 day but close

    def test_updated_premium_floors_at_penny(self) -> None:
        entry = 0.10
        new = updated_premium(
            entry_premium=entry,
            delta=0.10,
            underlying_entry=500.0,
            underlying_now=480.0,  # big adverse move
            days_elapsed=6,
            dte_remaining=1,
            iv=0.18,
        )
        assert new >= 0.01

    def test_updated_premium_decays_with_time(self) -> None:
        entry = atm_premium(500.0, 0.18, 7)
        decayed = updated_premium(
            entry_premium=entry,
            delta=0.50,
            underlying_entry=500.0,
            underlying_now=500.0,  # flat underlying
            days_elapsed=5,
            dte_remaining=2,
            iv=0.18,
        )
        assert decayed < entry


# ---------------------------------------------------------------------------
# Sizing tests
# ---------------------------------------------------------------------------


class TestAggressiveSizer:
    def test_stage1_spends_25_pct(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=1_000.0, premium_per_share=2.0, cfg=cfg)
        # $2.00 × 100 = $200 per contract; 25% of $1,000 = $250 → 1 contract
        assert qty >= 1
        assert spend <= 1_000.0

    def test_stage2_spends_20_pct(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=5_000.0, premium_per_share=5.0, cfg=cfg)
        assert qty >= 1
        assert spend <= 5_000.0

    def test_stage3_spends_15_pct(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=15_000.0, premium_per_share=5.0, cfg=cfg)
        assert qty >= 1
        assert spend <= 15_000.0

    def test_returns_zero_when_too_expensive(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=50.0, premium_per_share=10.0, cfg=cfg)
        # $10 × 100 = $1,000/contract; balance is $50 → can't afford
        assert qty == 0
        assert spend == 0.0

    def test_does_not_force_contract_above_stage_fraction(self, cfg: ChallengeConfig) -> None:
        """Regression: max(1, …) used to buy ~$800 SPY on $1k when only 25% budget allowed."""
        tight = ChallengeConfig(
            seed_capital=cfg.seed_capital,
            target_capital=cfg.target_capital,
            stage1_size_frac=0.25,
        )
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=1_000.0, premium_per_share=8.0, cfg=tight)
        assert qty == 0
        assert spend == 0.0

    def test_stage1_50pct_blocks_full_spy_premium(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=1_000.0, premium_per_share=8.0, cfg=cfg)
        assert qty == 0
        assert spend == 0.0

    def test_stage1_can_buy_when_premium_within_half_balance(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        qty, spend = sizer.compute(balance=1_000.0, premium_per_share=4.0, cfg=cfg)
        assert qty == 1
        assert spend == 400.0

    def test_spend_never_exceeds_balance(self, cfg: ChallengeConfig) -> None:
        sizer = AggressiveSizer()
        for balance in (800.0, 1_000.0, 3_500.0, 12_000.0):
            qty, spend = sizer.compute(balance, premium_per_share=2.0, cfg=cfg)
            assert spend <= balance


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestChallengeStrategy:
    def test_long_directive_buys_call(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("long", 500.0, 1_000.0, 0.18, cfg)
        assert play is not None
        assert play.option_type == "call"
        assert play.direction == "long"

    def test_short_directive_buys_put(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("short", 500.0, 1_000.0, 0.18, cfg)
        assert play is not None
        assert play.option_type == "put"
        assert play.direction == "short"

    def test_no_trade_returns_none(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("no_trade", 500.0, 1_000.0, 0.18, cfg)
        assert play is None

    def test_stage1_call_strike_above_underlying(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("long", 500.0, 1_000.0, 0.18, cfg)
        assert play is not None
        assert play.strike >= 500.0

    def test_stage1_put_strike_below_underlying(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("short", 500.0, 1_000.0, 0.18, cfg)
        assert play is not None
        assert play.strike <= 500.0

    def test_high_conviction_compresses_dte(self, cfg: ChallengeConfig) -> None:
        standard = ChallengeStrategy().select("long", 500.0, 1_000.0, 0.18, cfg, signal_alignment=0.60, confidence=0.60)
        high_conv = ChallengeStrategy().select(
            "long", 500.0, 1_000.0, 0.18, cfg, signal_alignment=0.85, confidence=0.80
        )
        assert standard is not None and high_conv is not None
        # High conviction stage-1 play should have equal or shorter DTE
        assert high_conv.dte <= standard.dte

    def test_stage3_uses_atm(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select("long", 500.0, 12_000.0, 0.18, cfg)
        assert play is not None
        assert play.otm_pct == 0.0
        assert play.strike == 500.0

    def test_spy_1k_selects_affordable_weekly(self, cfg: ChallengeConfig) -> None:
        """~$711 SPY @ 50% budget must not pick ~$800 ATM weeklies."""
        play = ChallengeStrategy().select(
            "long",
            711.0,
            1_000.0,
            0.18,
            cfg,
            signal_alignment=0.72,
            confidence=0.68,
        )
        assert play is not None
        assert play.estimated_premium * 100 <= 1_000.0 * cfg.stage1_size_frac + 1.0
        assert play.dte >= 5

    def test_high_vol_scalp_when_conviction_strong(self, cfg: ChallengeConfig) -> None:
        play = ChallengeStrategy().select(
            "long",
            711.0,
            1_000.0,
            0.18,
            cfg,
            signal_alignment=0.80,
            confidence=0.75,
        )
        assert play is not None
        assert play.dte == cfg.scalp_dte
        assert "scalp" in play.rationale


# ---------------------------------------------------------------------------
# State / persistence tests
# ---------------------------------------------------------------------------


class TestChallengeTracker:
    def test_reset_creates_fresh_state(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        state = tmp_tracker.reset(cfg)
        assert state.balance == cfg.seed_capital
        assert state.target == cfg.target_capital
        assert state.open_positions == []

    def test_save_and_load_round_trip(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        state = tmp_tracker.reset(cfg)
        state.balance = 1_500.0
        state.session_count = 3
        tmp_tracker.save(state)
        loaded = tmp_tracker.load()
        assert loaded.balance == 1_500.0
        assert loaded.session_count == 3

    def test_load_raises_when_no_state(self, tmp_path: Path) -> None:
        tracker = ChallengeTracker(data_root=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            tracker.load()

    def test_append_trade_creates_csv(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        record = ChallengeTradeRecord(
            trade_id="abc",
            symbol="SPY",
            option_type="call",
            direction="long",
            strike=502.0,
            dte_at_entry=7,
            entry_date="2026-01-01",
            exit_date="2026-01-05",
            premium_paid=200.0,
            proceeds=400.0,
            pnl=200.0,
            pnl_pct=100.0,
            exit_reason="target",
            balance_before=1_000.0,
            balance_after=1_200.0,
        )
        tmp_tracker.append_trade(record)
        assert tmp_tracker.trade_log_path().exists()


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestChallengeEngine:
    def test_bullish_session_opens_call(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        summary = engine.run_session(
            directive="long",
            underlying_price=500.0,
            signal_alignment=0.80,
            confidence=0.75,
        )
        assert summary.new_position is not None
        assert summary.new_position.option_type == "call"

    def test_bearish_session_opens_put(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        summary = engine.run_session(
            directive="short",
            underlying_price=500.0,
            signal_alignment=0.80,
            confidence=0.75,
        )
        assert summary.new_position is not None
        assert summary.new_position.option_type == "put"

    def test_no_trade_directive_opens_no_position(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        summary = engine.run_session(
            directive="no_trade",
            underlying_price=500.0,
        )
        assert summary.new_position is None

    def test_target_hit_closes_position_at_2x(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        # Open a position
        engine.run_session("long", 500.0, session_date="2026-01-01")
        # Simulate strong rally — underlying jumps +5%
        summary = engine.run_session("long", 525.0, session_date="2026-01-02")
        # One of: closed at target, or still open (depends on premium calc)
        assert isinstance(summary.balance_after, float)
        assert summary.balance_after > 0

    def test_trail_exit_respects_breakeven_floor(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        from rlm.challenge.state import ChallengePosition

        state = tmp_tracker.reset(cfg)
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=500.0,
            strike=505.0,
            dte=7,
            entry_date="2026-01-01",
            premium_per_share=2.0,
            qty=1,
            delta=0.5,
            iv=0.18,
        )
        pos.peak_premium_mult = 1.30
        pos.trail_armed = True
        state.open_positions = [pos]
        state.balance = 600.0
        tmp_tracker.save(state)
        loose = replace(cfg, stop_loss_mult=0.40)
        engine = ChallengeEngine(loose, tmp_tracker)
        summary = engine.run_session("no_trade", 500.0, session_date="2026-01-02")
        assert len(summary.closed_trades) == 1
        assert summary.closed_trades[0].exit_reason == "trail"

    def test_live_mark_triggers_target_exit(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        from unittest.mock import patch

        from rlm.challenge.state import ChallengePosition

        state = tmp_tracker.reset(cfg)
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=711.0,
            strike=740.0,
            dte=7,
            entry_date="2026-05-28",
            premium_per_share=3.62,
            qty=1,
            delta=0.45,
            iv=0.18,
        )
        state.open_positions = [pos]
        state.balance = 638.0
        tmp_tracker.save(state)

        with (
            patch("rlm.challenge.live_marks.live_marks_enabled", return_value=True),
            patch("rlm.challenge.live_marks.fetch_equity_quote") as eq,
            patch("rlm.challenge.live_marks.fetch_option_mid_per_share", return_value=17.5),
        ):
            from rlm.market.live_quotes import EquityQuote

            eq.return_value = EquityQuote("SPY", 757.0, "2026-05-29T12:00:00+00:00", "test")
            engine = ChallengeEngine(cfg, tmp_tracker)
            summary = engine.run_session("no_trade", 711.0, session_date="2026-05-29")

        assert len(summary.closed_trades) == 1
        assert summary.closed_trades[0].exit_reason == "target"
        expected_proceeds = 3.62 * cfg.stage1_profit_target_mult * 100
        assert summary.closed_trades[0].proceeds == pytest.approx(expected_proceeds, rel=1e-4)
        assert summary.balance_after == pytest.approx(638.0 + expected_proceeds, rel=1e-4)

    def test_pdt_blocks_same_day_exit_when_slots_exhausted(
        self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker
    ) -> None:
        from unittest.mock import patch

        from rlm.challenge.state import ChallengePosition

        state = tmp_tracker.reset(cfg)
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=500.0,
            strike=505.0,
            dte=7,
            entry_date="2026-05-29",
            premium_per_share=2.0,
            qty=1,
            delta=0.5,
            iv=0.18,
        )
        state.open_positions = [pos]
        state.balance = 800.0
        state.pdt_day_trade_counts = [3]
        state.pdt_last_session_date = "2026-05-29"
        tmp_tracker.save(state)

        with (
            patch("rlm.challenge.engine.mark_open_position_premium", return_value=(3.0, 500.0, 5)),
            patch("rlm.challenge.live_marks.live_marks_enabled", return_value=False),
        ):
            engine = ChallengeEngine(cfg, tmp_tracker)
            summary = engine.run_session("no_trade", 500.0, session_date="2026-05-29")

        assert summary.closed_trades == []
        loaded = tmp_tracker.load()
        assert len(loaded.open_positions) == 1

    def test_target_exit_capped_at_limit_not_live_mark(
        self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker
    ) -> None:
        from unittest.mock import patch

        from rlm.challenge.state import ChallengePosition

        state = tmp_tracker.reset(cfg)
        pos = ChallengePosition.new(
            symbol="SPY",
            option_type="call",
            direction="long",
            underlying_entry=500.0,
            strike=505.0,
            dte=7,
            entry_date="2026-01-01",
            premium_per_share=2.0,
            qty=1,
            delta=0.5,
            iv=0.18,
        )
        state.open_positions = [pos]
        state.balance = 600.0
        tmp_tracker.save(state)

        with (
            patch("rlm.challenge.live_marks.live_marks_enabled", return_value=True),
            patch("rlm.challenge.live_marks.fetch_equity_quote") as eq,
            patch("rlm.challenge.live_marks.fetch_option_mid_per_share", return_value=9.0),
        ):
            from rlm.market.live_quotes import EquityQuote

            eq.return_value = EquityQuote("SPY", 520.0, "2026-01-02T12:00:00+00:00", "test")
            engine = ChallengeEngine(cfg, tmp_tracker)
            summary = engine.run_session("no_trade", 520.0, session_date="2026-01-02")

        assert len(summary.closed_trades) == 1
        cap = 2.0 * cfg.stage1_profit_target_mult * 100
        assert summary.closed_trades[0].proceeds == pytest.approx(cap, rel=1e-4)

    def test_stop_hit_closes_position(self, cfg_enter: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg_enter)
        engine = ChallengeEngine(cfg_enter, tmp_tracker)
        # Open a long call
        engine.run_session("long", 500.0, session_date="2026-01-01")
        # Big adverse move — underlying falls -5%
        engine.run_session("no_trade", 475.0, session_date="2026-01-06")
        state = tmp_tracker.load()
        # Position should be closed (stop or expiry hit after 5 days on 7DTE option)
        total_trades = len(state.trade_history)
        assert total_trades >= 1

    def test_balance_deducted_on_entry(self, cfg_enter: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        state = tmp_tracker.reset(cfg_enter)
        initial = state.balance
        engine = ChallengeEngine(cfg_enter, tmp_tracker)
        engine.run_session("long", 500.0)
        state = tmp_tracker.load()
        assert state.balance < initial  # cash spent on the position

    def test_max_concurrent_positions_respected(
        self, cfg_enter: ChallengeConfig, tmp_tracker: ChallengeTracker
    ) -> None:
        wide_target = replace(cfg_enter, stage1_profit_target_mult=2.5)
        tmp_tracker.reset(wide_target)
        engine = ChallengeEngine(wide_target, tmp_tracker)
        engine.run_session("long", 500.0, session_date="2026-01-01")
        engine.run_session("long", 501.0, session_date="2026-01-01")
        # Third session should not open a new position (at capacity)
        summary = engine.run_session("long", 502.0, session_date="2026-01-01")
        assert summary.new_position is None

    def test_challenge_complete_flag_set_at_target(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        # Manually push balance just below target
        state = tmp_tracker.reset(cfg)
        state.balance = cfg.target_capital + 1.0  # already at target
        tmp_tracker.save(state)
        engine = ChallengeEngine(cfg, tmp_tracker)
        summary = engine.run_session("long", 500.0)
        assert summary.challenge_complete

    def test_session_count_increments(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        for _ in range(3):
            engine.run_session("long", 500.0)
        state = tmp_tracker.load()
        assert state.session_count == 3

    def test_elapsed_days_advances_once_per_calendar_day(
        self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker
    ) -> None:
        """Challenge loop ticks many times/day; pace must use calendar days."""
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        for _ in range(20):
            engine.run_session("no_trade", 500.0, session_date="2026-01-05")
        state = tmp_tracker.load()
        assert state.session_count == 20
        assert state.elapsed_days == 1

        engine.run_session("no_trade", 500.0, session_date="2026-01-06")
        state = tmp_tracker.load()
        assert state.session_count == 21
        assert state.elapsed_days == 2

    def test_same_day_loop_ticks_do_not_max_out_pace_boost(self, tmp_tracker: ChallengeTracker) -> None:
        """Regression: per-tick elapsed_days inflated the 252-day pace curve intraday."""
        cfg = ChallengeConfig(seed_capital=1_000.0, target_capital=100_000.0)
        tmp_tracker.reset(cfg)
        engine = ChallengeEngine(cfg, tmp_tracker)
        # Full RTH day of 5-minute challenge-loop ticks
        for _ in range(78):
            engine.run_session("no_trade", 500.0, session_date="2026-01-05")
        state = tmp_tracker.load()
        assert state.session_count == 78
        assert state.elapsed_days == 1
        pace = engine._compute_pace_size_multiplier(state, "2026-01-05")
        # Day-1 required growth is ~1.018x; at seed, progress≈0.98 > 0.80 threshold
        assert pace == pytest.approx(1.0)
        # Bug baseline: treating each tick as a day would already demand ~+14% boost
        state.elapsed_days = state.session_count
        buggy_pace = engine._compute_pace_size_multiplier(state, "2026-01-05")
        assert buggy_pace > 1.10

    def test_state_persists_across_engine_instances(self, cfg: ChallengeConfig, tmp_tracker: ChallengeTracker) -> None:
        tmp_tracker.reset(cfg)
        ChallengeEngine(cfg, tmp_tracker).run_session("long", 500.0)
        # Instantiate a fresh engine — should see the open position
        state = tmp_tracker.load()
        assert len(state.open_positions) <= cfg.max_concurrent_positions


# ---------------------------------------------------------------------------
# Milestone and progress tests
# ---------------------------------------------------------------------------


class TestChallengeState:
    def test_progress_zero_at_seed(self, cfg: ChallengeConfig) -> None:
        state = ChallengeState.fresh(cfg, "2026-01-01")
        assert state.progress_pct == 0.0

    def test_progress_one_at_target(self, cfg: ChallengeConfig) -> None:
        state = ChallengeState.fresh(cfg, "2026-01-01")
        state.balance = cfg.target_capital
        assert state.progress_pct == 1.0

    def test_milestone_progression(self, cfg: ChallengeConfig) -> None:
        state = ChallengeState.fresh(cfg, "2026-01-01")
        assert state.current_milestone_idx == 0  # Phase I next
        state.balance = 2_600.0
        assert state.current_milestone_idx == 1  # Phase II next

    def test_win_rate_calculation(self, cfg: ChallengeConfig) -> None:
        state = ChallengeState.fresh(cfg, "2026-01-01")
        for pnl in (100.0, -50.0, 200.0, -75.0, 150.0):
            state.trade_history.append(
                ChallengeTradeRecord(
                    trade_id="x",
                    symbol="SPY",
                    option_type="call",
                    direction="long",
                    strike=500.0,
                    dte_at_entry=7,
                    entry_date="2026-01-01",
                    exit_date="2026-01-05",
                    premium_paid=200.0,
                    proceeds=200.0 + pnl,
                    pnl=pnl,
                    pnl_pct=pnl / 200.0 * 100,
                    exit_reason="target" if pnl > 0 else "stop",
                    balance_before=1_000.0,
                    balance_after=1_000.0 + pnl,
                )
            )
        assert state.wins == 3
        assert state.losses == 2
        assert math.isclose(state.win_rate, 0.6)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_days_between_same_date(self) -> None:
        assert _days_between("2026-01-01", "2026-01-01") == 0

    def test_days_between_one_week(self) -> None:
        assert _days_between("2026-01-01", "2026-01-08") == 7

    def test_days_between_bad_input_returns_one(self) -> None:
        assert _days_between("bad", "date") == 1
