"""Typed models for the challenge engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageSizingRule:
    """Premium outlay and max-loss limits for a given equity stage."""
    equity_min: float
    equity_max: float
    premium_outlay_pct: float   # fraction of equity to spend on premium
    max_loss_pct: float         # max tolerated loss per trade as % of equity


@dataclass
class ChallengePipelineConfig:
    """Pipeline-layer parameters: scoring, trade-mode gating, and contract preferences.

    Distinct from ``rlm.challenge.config.ChallengeConfig`` which owns execution
    parameters (exit multipliers, DTE per stage, position sizing fractions).
    """

    # Account targets
    starting_equity: float = 1_000.0
    target_equity: float = 25_000.0

    # Universe — challenge only; keep narrow and liquid
    allowed_universe: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "META"
    ])

    # Setup quality gates
    min_setup_score: float = 0.55        # below this → no_trade
    elite_setup_score: float = 0.70      # above this → scalp eligible (was 0.78 — too restrictive)
    top_setups_per_session: int = 2      # max entries per session

    # PDT
    max_day_trades_per_5d: int = 3       # conservative PDT ceiling

    # Stage sizing tiers
    stage_sizing: list[StageSizingRule] = field(default_factory=lambda: [
        StageSizingRule(1_000,  2_500, premium_outlay_pct=0.12, max_loss_pct=0.025),
        StageSizingRule(2_500,  7_500, premium_outlay_pct=0.15, max_loss_pct=0.030),
        StageSizingRule(7_500, 25_000, premium_outlay_pct=0.18, max_loss_pct=0.035),
    ])

    # Stop / trailing stop
    hard_stop_pct: float = 0.28          # -28% of premium paid
    trail_activate_pct: float = 0.18     # activate trail after +18% gain
    trail_drawdown_pct: float = 0.12     # trail permits -12% from peak
    profit_target_pct: float = 0.22      # first profit target
    partial_take_pct: float = 0.50       # take 50% at first target

    # Contract preferences (swing)
    swing_delta_min: float = 0.45
    swing_delta_max: float = 0.65
    swing_dte_min: int = 7
    swing_dte_max: int = 21

    # Contract preferences (scalp)
    scalp_delta_min: float = 0.40        # ATM-ish; matches engine's 0DTE ATM play
    scalp_delta_max: float = 0.60
    scalp_dte_min: int = 0               # true 0DTE allowed (was 3)
    scalp_dte_max: int = 3               # short-dated cap (was 10)

    # Max bid-ask spread as fraction of mid-price
    max_spread_pct: float = 0.06

    # Setup score weights (must sum to 1.0)
    weight_seven_confidence: float = 0.35
    weight_signal_alignment: float = 0.25
    weight_historical_edge: float = 0.20
    weight_dealer_support: float = 0.20


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------

@dataclass
class ChallengeAccountState:
    """Mutable account state persisted to data/challenge/state.json."""
    current_equity: float = 1_000.0
    peak_equity: float = 1_000.0
    milestone_stage: int = 1            # 1, 2, or 3 matching stage_sizing tiers
    open_positions_count: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sessions_run: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses


# ---------------------------------------------------------------------------
# PDT tracker
# ---------------------------------------------------------------------------

@dataclass
class PDTTracker:
    """Track intraday round-trips for PDT compliance simulation."""
    day_trades_used_last_5d: list[int] = field(default_factory=list)
    # Each element = count of day-trades used on that calendar date (last 5)

    @property
    def day_trades_remaining(self) -> int:
        used = sum(self.day_trades_used_last_5d[-5:])
        return max(0, 3 - used)

    @property
    def same_day_exit_allowed(self) -> bool:
        return self.day_trades_remaining > 0

    @property
    def must_hold_overnight_if_entered(self) -> bool:
        return not self.same_day_exit_allowed

    def record_day_trade(self) -> None:
        """Record one round-trip intraday."""
        if self.day_trades_used_last_5d:
            self.day_trades_used_last_5d[-1] += 1
        else:
            self.day_trades_used_last_5d.append(1)

    def new_session(self) -> None:
        """Call at start of each trading day."""
        self.day_trades_used_last_5d.append(0)
        if len(self.day_trades_used_last_5d) > 5:
            self.day_trades_used_last_5d = self.day_trades_used_last_5d[-5:]


# ---------------------------------------------------------------------------
# Derived decision types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SetupScoreResult:
    """Setup ranking output."""
    setup_score: float
    conviction: Literal["low", "medium", "high", "elite"]
    passed_threshold: bool
    ranking_reason: str


@dataclass(frozen=True)
class TradeModeDecision:
    """Determines scalp vs swing_candidate vs no_trade."""
    trade_mode: Literal["scalp", "swing_candidate", "no_trade"]
    same_day_exit_allowed: bool
    pdt_reason: str


@dataclass(frozen=True)
class ContractProfileRecommendation:
    """Lightweight contract selection guidance."""
    target_delta_min: float
    target_delta_max: float
    preferred_dte_min: int
    preferred_dte_max: int
    max_spread_pct: float
    liquidity_tier: Literal["high", "medium", "low"]
    note: str


@dataclass(frozen=True)
class RiskPlan:
    """Complete risk plan for a single challenge trade."""
    premium_outlay_pct: float
    max_account_loss_pct: float
    hard_stop_pct: float
    trail_activate_pct: float
    trail_drawdown_pct: float
    profit_target_pct: float
    partial_take_pct: float
    use_underlying_invalidation: bool
    force_close_dte_threshold: int


@dataclass(frozen=True)
class ChallengeDirective:
    """Final challenge decision artifact."""
    symbol: str
    setup_score: float
    conviction: Literal["low", "medium", "high", "elite"]
    directive: Literal["long", "short", "no_trade"]
    trade_mode: Literal["scalp", "swing_candidate", "no_trade"]
    same_day_exit_allowed: bool
    pdt_slots_remaining: int
    contract_profile: ContractProfileRecommendation
    risk_plan: RiskPlan
    reason_summary: str
