"""ChallengeEngine — orchestrates one dry-run session of the $1K→$25K challenge.

Each call to ``run_session`` does two things in order:
  1. Evaluate all open positions: check exit conditions, close as needed.
  2. Evaluate a new entry: if a slot is available and the persona agrees, open one.

State is loaded from / saved to ``ChallengeTracker`` at the boundaries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.pricing import updated_premium
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
    """Dry-run session runner for the aggressive options challenge.

    Parameters
    ----------
    cfg:
        Challenge configuration.
    tracker:
        Persistence layer (handles load/save of ``ChallengeState``).
    """

    def __init__(
        self,
        cfg: ChallengeConfig,
        tracker: ChallengeTracker,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self._sizer = AggressiveSizer()
        self._strategy = ChallengeStrategy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_session(
        self,
        directive: Literal["long", "short", "no_trade"],
        underlying_price: float,
        *,
        signal_alignment: float = 0.7,
        confidence: float = 0.7,
        iv: float | None = None,
        session_date: str | None = None,
    ) -> SessionSummary:
        """Execute one challenge session.

        Parameters
        ----------
        directive:
            From ``PersonaDecisionPipeline`` → ``sisko.directive``.
        underlying_price:
            Current market price of the underlying (e.g. SPY last trade).
        signal_alignment:
            From ``seven.signal_alignment``; used by strategy for conviction gating.
        confidence:
            From ``seven.confidence``; used by strategy for conviction gating.
        iv:
            Implied volatility override.  Defaults to ``cfg.default_iv``.
        session_date:
            ISO date string (``YYYY-MM-DD``).  Defaults to today (UTC).
        """
        iv = iv or self.cfg.default_iv
        session_date = session_date or date.today().isoformat()
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        state = self.tracker.load()
        balance_before = state.balance
        prev_milestone_idx = state.current_milestone_idx

        closed_trades: list[ChallengeTradeRecord] = []

        # 1. Evaluate open positions
        for pos in list(state.open_positions):
            record = self._evaluate_position(pos, underlying_price, iv, session_date, state)
            if record is not None:
                closed_trades.append(record)

        # 2. Consider new entry (if slots available and challenge not yet complete)
        new_position: ChallengePosition | None = None
        if state.balance < self.cfg.target_capital and len(state.open_positions) < self.cfg.max_concurrent_positions:
            play = self._strategy.select(
                directive,
                underlying_price,
                state.balance,
                iv,
                self.cfg,
                signal_alignment=signal_alignment,
                confidence=confidence,
            )
            if play is not None:
                qty, spend = self._sizer.compute(state.balance, play.estimated_premium, self.cfg)
                if qty > 0 and spend <= state.balance:
                    pos = ChallengePosition.new(
                        symbol=self.cfg.symbol,
                        option_type=play.option_type,
                        direction=play.direction,
                        underlying_entry=underlying_price,
                        strike=play.strike,
                        dte=play.dte,
                        entry_date=session_date,
                        premium_per_share=play.estimated_premium,
                        qty=qty,
                        delta=play.estimated_delta,
                        iv=iv,
                    )
                    state.open_positions.append(pos)
                    state.balance -= spend
                    new_position = pos

        state.session_count += 1
        state.last_updated = now_iso
        self.tracker.save(state)

        # Detect milestone clears
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
            message=_compose_message(state, closed_trades, new_position, challenge_complete),
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
        days_elapsed = _days_between(pos.entry_date, session_date)
        new_dte = max(0, pos.dte_at_entry - days_elapsed)
        new_premium = updated_premium(
            entry_premium=pos.premium_per_share,
            delta=pos.delta_at_entry,
            underlying_entry=pos.underlying_entry,
            underlying_now=underlying_now,
            days_elapsed=days_elapsed,
            dte_remaining=new_dte,
            iv=iv,
        )

        pos.dte_remaining = new_dte
        pos.current_premium = new_premium
        pos.current_value = new_premium * pos.qty * 100
        pos.unrealised_pnl = pos.current_value - pos.total_cost

        # Determine exit condition
        mult = new_premium / pos.premium_per_share
        exit_reason: Literal["target", "stop", "expiry", "manual"] | None = None

        if mult >= self.cfg.profit_target_mult:
            exit_reason = "target"
        elif mult <= self.cfg.stop_loss_mult:
            exit_reason = "stop"
        elif new_dte <= self.cfg.min_dte_exit:
            exit_reason = "expiry"

        if exit_reason is None:
            return None  # hold

        # Close the position
        proceeds = pos.current_value
        pnl = proceeds - pos.total_cost
        balance_before = state.balance
        state.balance = state.balance + proceeds
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            f"×{new_pos.qty} @ ${new_pos.premium_per_share:.2f} (${new_pos.total_cost:.2f} spend)"
        )
    if not parts:
        parts.append("No action this session.")
    if complete:
        parts.append("🏁 CHALLENGE COMPLETE — $25,000 target reached!")
    return "  ".join(parts)
