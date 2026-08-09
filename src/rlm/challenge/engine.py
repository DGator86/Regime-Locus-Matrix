"""ChallengeEngine -- orchestrates one dry-run session of the $1K->$100K challenge.

Each call to ``run_session`` does two things in order:
  1. Evaluate all open positions: check exit conditions, close as needed.
  2. Evaluate a new entry: if a slot is available and the persona agrees, open one.

State is loaded from / saved to ``ChallengeTracker`` at the boundaries.

Compliant with FINRA Rule 4210 (amended), effective June 4, 2026.  The challenge
uses a cash account (long options, fully paid).  Intraday exposure is tracked on
every open/close; new entries are blocked when proposed exposure would exceed
balance × intraday_exposure_limit_frac (default 1.0 for cash accounts).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.live_marks import mark_open_position_premium
from rlm.challenge.pricing import entry_friction, exit_friction, min_tick_round
from rlm.challenge.sizing import AggressiveSizer
from rlm.challenge.state import (
    ChallengePosition,
    ChallengeState,
    ChallengeTradeRecord,
)
from rlm.challenge.strategy import ChallengeStrategy
from rlm.challenge.tracker import ChallengeTracker


@dataclass
class SessionSummary:
    """Result bundle returned by one ``ChallengeEngine.run_session`` call."""

    session_date: str
    directive: str
    balance_before: float
    balance_after: float
    closed_trades: list[ChallengeTradeRecord] = field(default_factory=list)
    new_position: ChallengePosition | None = None
    milestone_cleared: str | None = None
    challenge_complete: bool = False
    message: str = ""


class ChallengeEngine:
    """Dry-run session runner for the aggressive options challenge."""

    def __init__(self, cfg: ChallengeConfig, tracker: ChallengeTracker) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self._sizer = AggressiveSizer()
        self._strategy = ChallengeStrategy()

    def run_session(
        self,
        directive: Literal["long", "short", "no_trade"],
        underlying_price: float,
        *,
        signal_alignment: float = 0.7,
        confidence: float = 0.7,
        iv: float | None = None,
        realized_vol: float | None = None,
        session_date: str | None = None,
        regime_key: str = "",
    ) -> SessionSummary:
        """Execute one challenge session."""
        if not iv:
            if realized_vol:
                iv = realized_vol * (1.0 + self.cfg.iv_vol_premium)
            else:
                iv = self.cfg.default_iv
        session_date = session_date or date.today().isoformat()
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        state = self.tracker.load()
        balance_before = state.balance
        prev_milestone_idx = state.current_milestone_idx

        # Set start_date on first session
        if state.start_date == "":
            state.start_date = session_date

        # Increment elapsed_days each session
        state.elapsed_days += 1

        # Reset daily P&L tracker at the start of a new calendar day
        if state.daily_pnl_date != session_date:
            state.daily_realized_pnl = 0.0
            state.daily_pnl_date = session_date

        closed_trades: list[ChallengeTradeRecord] = []

        # 1. Evaluate open positions
        for pos in list(state.open_positions):
            # When session spot is unavailable (<=0), mark from each leg's entry
            # underlying so a placeholder price cannot false-stop / false-target.
            mark_px = (
                float(underlying_price)
                if underlying_price and underlying_price > 0
                else float(pos.underlying_entry or 0.0)
            )
            record = self._evaluate_position(pos, mark_px, iv, session_date, state)
            if record is not None:
                closed_trades.append(record)
                state.daily_realized_pnl += record.pnl

        # Recalculate intraday exposure after close evaluations
        state.intraday_exposure = sum(p.current_value for p in state.open_positions)
        if state.intraday_exposure > state.intraday_exposure_peak:
            state.intraday_exposure_peak = state.intraday_exposure

        # 2. Consider new entry
        new_position: ChallengePosition | None = None
        entry_skip_reason: str | None = None
        if len(state.open_positions) >= self.cfg.max_concurrent_positions:
            entry_skip_reason = (
                f"Holding {len(state.open_positions)} open leg(s); "
                f"cash ${state.balance:,.2f}"
            )
        elif state.balance < self.cfg.target_capital and len(state.open_positions) < self.cfg.max_concurrent_positions:
            if not entry_skip_reason and (not underlying_price or underlying_price <= 0):
                entry_skip_reason = "underlying_price_unavailable"

            if not entry_skip_reason and _daily_loss_limit_reached(state.daily_realized_pnl, state.balance, self.cfg):
                entry_skip_reason = "daily_loss_limit_reached"

            if not entry_skip_reason and _is_event_blackout(session_date, self.cfg):
                entry_skip_reason = "major_event_blackout"

            if not entry_skip_reason and not _regime_win_rate_ok(regime_key, state, self.cfg):
                entry_skip_reason = "regime_win_rate_below_threshold"

            if not entry_skip_reason:
                play = self._strategy.select(
                    directive,
                    underlying_price,
                    state.balance,
                    iv,
                    self.cfg,
                    signal_alignment=signal_alignment,
                    confidence=confidence,
                )
                if play is None:
                    entry_skip_reason = (
                        f"No affordable weekly/scalp under "
                        f"${state.balance * self.cfg.size_fraction(state.balance):,.0f} premium budget"
                    )
                else:
                    proposed_exposure = state.intraday_exposure + (play.estimated_premium * 1 * 100)
                    exposure_limit = state.balance * self.cfg.intraday_exposure_limit_frac
                    if proposed_exposure > exposure_limit:
                        entry_skip_reason = (
                            f"intraday_exposure_limit: proposed ${proposed_exposure:,.0f} "
                            f"would exceed limit ${exposure_limit:,.0f} "
                            f"({self.cfg.intraday_exposure_limit_frac:.0%} of ${state.balance:,.0f})"
                        )
                    same_dir_spend = sum(
                        p.total_cost for p in state.open_positions if p.direction == play.direction
                    )
                    if not entry_skip_reason and same_dir_spend / max(state.balance, 1.0) >= self.cfg.max_same_direction_premium_frac:
                        entry_skip_reason = "correlation_exposure_limit"
                    if not entry_skip_reason:
                        pace_mult = self._compute_pace_size_multiplier(state, session_date)
                        budget_override = state.balance * self.cfg.size_fraction(state.balance) * pace_mult
                        qty, spend = self._sizer.compute(
                            state.balance,
                            play.estimated_premium,
                            self.cfg,
                            budget_override=budget_override,
                        )
                        # Safety: never spend more than real balance
                        if spend > state.balance:
                            qty, spend = self._sizer.compute(state.balance, play.estimated_premium, self.cfg)
                        if qty <= 0 or spend > state.balance:
                            cost = float(play.estimated_premium) * 100.0
                            entry_skip_reason = (
                                f"insufficient cash for 1 contract "
                                f"(need ~${cost:,.0f}, balance ${state.balance:,.2f})"
                            )
                        if qty > 0 and spend <= state.balance:
                            e_friction = entry_friction(play.estimated_premium, qty, self.cfg)
                            total_entry_cost = spend + e_friction
                            if total_entry_cost > state.balance:
                                total_entry_cost = spend
                            pos = ChallengePosition.new(
                                symbol=self.cfg.symbol,
                                option_type=play.option_type,
                                direction=play.direction,
                                underlying_entry=underlying_price,
                                strike=play.strike,
                                dte=play.dte,
                                entry_date=session_date,
                                premium_per_share=min_tick_round(play.estimated_premium),
                                qty=qty,
                                delta=play.estimated_delta,
                                iv=iv,
                                regime_key=regime_key,
                            )
                            state.open_positions.append(pos)
                            state.balance -= total_entry_cost
                            new_position = pos
                            # Update intraday exposure after entry
                            state.intraday_exposure = sum(p.current_value for p in state.open_positions)
                            if state.intraday_exposure > state.intraday_exposure_peak:
                                state.intraday_exposure_peak = state.intraday_exposure

        # End-of-day margin call simulation (FINRA Rule 4210 Path B compliance)
        if self.cfg.end_of_day_margin_call_enabled:
            eod_exposure = sum(p.current_value for p in state.open_positions)
            eod_limit = state.balance * self.cfg.intraday_exposure_limit_frac
            if eod_exposure > eod_limit:
                state.intraday_margin_calls += 1
                # Force-close the most expensive position to bring exposure under limit
                if state.open_positions:
                    worst = max(state.open_positions, key=lambda p: p.current_value)
                    record = self._evaluate_position(worst, underlying_price, iv, session_date, state)
                    if record:
                        closed_trades.append(record)

        state.session_count += 1
        state.last_updated = now_iso
        self.tracker.save(state)

        milestone_cleared: str | None = None
        if state.current_milestone_idx > prev_milestone_idx or state.balance >= self.cfg.target_capital:
            from rlm.challenge.config import MILESTONES

            idx = min(prev_milestone_idx, len(MILESTONES) - 1)
            milestone_cleared = MILESTONES[idx].label

        challenge_complete = state.balance >= self.cfg.target_capital

        return SessionSummary(
            session_date=session_date,
            directive=directive,
            balance_before=balance_before,
            balance_after=state.balance,
            closed_trades=closed_trades,
            new_position=new_position,
            milestone_cleared=milestone_cleared,
            challenge_complete=challenge_complete,
            message=_compose_message(
                state,
                closed_trades,
                new_position,
                challenge_complete,
                entry_skip_reason=entry_skip_reason,
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_position(
        self,
        pos: ChallengePosition,
        underlying_now: float,
        iv: float,
        session_date: str,
        state: ChallengeState,
    ) -> ChallengeTradeRecord | None:
        """Update position value; close and record if an exit condition is met."""
        new_premium, _, new_dte = mark_open_position_premium(
            pos,
            self.cfg,
            session_date=session_date,
            pipeline_underlying=underlying_now,
            iv=iv,
        )

        pos.dte_remaining = new_dte
        pos.current_premium = new_premium
        pos.current_value = new_premium * pos.qty * 100
        pos.unrealised_pnl = pos.current_value - pos.total_cost

        mult = new_premium / pos.premium_per_share
        if mult > pos.peak_premium_mult:
            pos.peak_premium_mult = mult
        trail_arm = self.cfg.trail_activate_for(state.balance)
        profit_target = self.cfg.profit_target_for(state.balance)
        if mult >= trail_arm:
            pos.trail_armed = True

        exit_reason: Literal["target", "stop", "trail", "expiry", "manual"] | None = None

        if mult >= profit_target:
            exit_reason = "target"
        elif mult <= self.cfg.stop_loss_for(state.balance):
            exit_reason = "stop"
        elif pos.trail_armed:
            trail_floor = max(
                self.cfg.min_trail_exit_mult,
                pos.peak_premium_mult * (1.0 - self.cfg.trail_retrace_frac),
            )
            if mult < trail_floor:
                exit_reason = "trail"
        elif new_dte <= self.cfg.min_dte_exit:
            exit_reason = "expiry"

        if exit_reason is None:
            return None

        fill_mult = self._exit_fill_mult(pos, exit_reason, state, mult)
        fill_premium = min_tick_round(pos.premium_per_share * fill_mult)
        raw_proceeds = fill_premium * pos.qty * 100
        e_friction = exit_friction(fill_premium, pos.qty, self.cfg)
        proceeds = max(raw_proceeds - e_friction, 0.0)
        pnl = proceeds - pos.total_cost
        balance_before = state.balance
        state.balance = state.balance + proceeds

        rk = getattr(pos, "regime_key", "")
        if rk:
            wins_list = state.regime_win_rates.setdefault(rk, [])
            wins_list.append(pnl > 0)
            if len(wins_list) > 20:
                state.regime_win_rates[rk] = wins_list[-20:]
        state.open_positions.remove(pos)
        pos.status = "closed"

        record = ChallengeTradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            symbol=pos.symbol,
            option_type=pos.option_type,
            direction=pos.direction,
            strike=pos.strike,
            dte_at_entry=pos.dte_at_entry,
            entry_date=pos.entry_date,
            exit_date=session_date,
            premium_paid=pos.total_cost,
            proceeds=proceeds,
            pnl=pnl,
            pnl_pct=pnl / pos.total_cost * 100.0,
            exit_reason=exit_reason,
            balance_before=balance_before,
            balance_after=state.balance,
        )
        state.trade_history.append(record)
        self.tracker.append_trade(record)
        return record

    def _exit_fill_mult(
        self,
        pos: ChallengePosition,
        exit_reason: Literal["target", "stop", "trail", "expiry", "manual"],
        state: ChallengeState,
        mark_mult: float,
    ) -> float:
        if exit_reason == "target":
            return self.cfg.profit_target_for(state.balance)
        if exit_reason == "stop":
            return self.cfg.stop_loss_for(state.balance)
        if exit_reason == "trail":
            return max(
                self.cfg.min_trail_exit_mult,
                pos.peak_premium_mult * (1.0 - self.cfg.trail_retrace_frac),
            )
        return mark_mult

    def _compute_pace_size_multiplier(self, state: ChallengeState, session_date: str) -> float:
        """Adjust sizing based on whether we are ahead or behind the 100x-in-12-month pace.

        progress_ratio = (balance / seed) / required_growth
        where required_growth = (target / seed) ^ (elapsed_days / challenge_days)
        """
        cfg = self.cfg
        if state.elapsed_days <= 0 or cfg.challenge_days <= 0:
            return 1.0
        elapsed = min(state.elapsed_days, cfg.challenge_days)
        frac = elapsed / cfg.challenge_days
        required_growth = (cfg.target_capital / cfg.seed_capital) ** frac
        actual_growth = state.balance / cfg.seed_capital
        if required_growth <= 0:
            return 1.0
        progress_ratio = actual_growth / required_growth
        if progress_ratio < cfg.pace_boost_threshold:
            shortfall = (cfg.pace_boost_threshold - progress_ratio) / cfg.pace_boost_threshold
            boost = min(shortfall, 1.0) * cfg.pace_boost_max
            return 1.0 + boost
        if progress_ratio > cfg.pace_reduce_threshold:
            excess = (progress_ratio - cfg.pace_reduce_threshold) / cfg.pace_reduce_threshold
            reduction = min(excess, 1.0) * cfg.pace_reduce_max
            return 1.0 - reduction
        return 1.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_event_blackout(session_date: str, cfg: ChallengeConfig) -> bool:
    if not cfg.major_event_dates:
        return False
    try:
        sd = date.fromisoformat(session_date)
    except ValueError:
        return False
    buffer_days = max(1, cfg.block_hours_before_event // 24)
    for event_str in cfg.major_event_dates:
        try:
            ev = date.fromisoformat(event_str)
        except ValueError:
            continue
        if 0 <= (ev - sd).days <= buffer_days:
            return True
    return False


def _regime_win_rate_ok(regime_key: str, state: ChallengeState, cfg: ChallengeConfig) -> bool:
    if not regime_key:
        return True
    wins_list = state.regime_win_rates.get(regime_key, [])
    if len(wins_list) < cfg.regime_win_rate_min_samples:
        return True
    win_rate = sum(wins_list) / len(wins_list)
    return win_rate >= cfg.regime_win_rate_min


def _daily_loss_limit_reached(daily_realized_pnl: float, balance: float, cfg: ChallengeConfig) -> bool:
    if daily_realized_pnl >= 0:
        return False
    limit_frac = cfg.max_daily_loss_frac(balance)
    return abs(daily_realized_pnl) / max(balance, 1.0) >= limit_frac


def _days_between(start: str, end: str) -> int:
    try:
        d1 = date.fromisoformat(start)
        d2 = date.fromisoformat(end)
        return max(0, (d2 - d1).days)
    except ValueError:
        return 1


def _compose_message(
    state: ChallengeState,
    closed: list,
    new_pos: ChallengePosition | None,
    complete: bool,
    *,
    entry_skip_reason: str | None = None,
) -> str:
    parts: list[str] = []
    for t in closed:
        sign = "+" if t.pnl >= 0 else ""
        parts.append(
            f"[{t.exit_reason.upper()}] {t.option_type.upper()} ${t.strike:.0f} "
            f"P&L {sign}${t.pnl:.2f} ({sign}{t.pnl_pct:.1f}%)"
        )
    if new_pos:
        parts.append(
            f"[ENTER] {new_pos.option_type.upper()} ${new_pos.strike:.0f} "
            f"x{new_pos.qty} @ ${new_pos.premium_per_share:.2f} (${new_pos.total_cost:.2f} spend)"
        )
    if not parts:
        if entry_skip_reason:
            parts.append(f"No entry: {entry_skip_reason}")
        elif state.open_positions:
            parts.append(
                f"Holding {len(state.open_positions)} open leg(s); cash ${state.balance:,.2f}"
            )
        else:
            parts.append("No action this session.")
    if complete:
        parts.append(
            f"CHALLENGE COMPLETE -- $100,000 growth target reached in {state.elapsed_days} trading sessions!"
        )
    return "  ".join(parts)
