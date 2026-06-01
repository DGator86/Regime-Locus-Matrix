"""ChallengeConfig — parameters for the $1K→$25K dry-run options challenge."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ChallengeMilestone:
    target: float
    label: str
    description: str


# Fixed progression checkpoints
MILESTONES: tuple[ChallengeMilestone, ...] = (
    ChallengeMilestone(2_500.0, "Phase I — Foundation", "Prove the edge; 2.5x from seed"),
    ChallengeMilestone(5_000.0, "Phase II — Build", "Compound the gains; 2x from Phase I"),
    ChallengeMilestone(10_000.0, "Phase III — Scale", "Increase size; 2x from Phase II"),
    ChallengeMilestone(25_000.0, "Phase IV — Arrival", "PDT threshold cleared"),
)


@dataclass(frozen=True)
class ChallengeConfig:
    """Full configuration for one challenge run.

    Defaults are tuned for an aggressive small-account growth strategy.
    Override only the knobs you need.
    """

    # ---- Account parameters -------------------------------------------------
    seed_capital: float = 1_000.0
    target_capital: float = 25_000.0
    symbol: str = "SPY"

    # ---- Position management ------------------------------------------------
    max_concurrent_positions: int = 1
    """Never hold more than this many open option positions simultaneously."""

    # ---- Sizing by account stage (fraction of balance per trade) ------------
    stage1_size_frac: float = 0.30
    """$1K – $3K: up to 30% of balance in premium per entry.  Reduced from 50% —
    at $1K a 50% allocation means one -18% stop wipes ~5.4% of the account; 30% keeps
    single-trade drawdown manageable even on a full stop."""
    stage2_size_frac: float = 0.20
    """$3K – $10K: 20% of balance in premium per trade."""
    stage3_size_frac: float = 0.15
    """$10K – $25K: 15% of balance in premium per trade."""

    # ---- Exit rules ---------------------------------------------------------
    stage3_profit_target_mult: float = 2.0
    """Stage 3 ($10K+): full take-profit multiple."""
    stage1_profit_target_mult: float = 1.38
    """Stage 1 ($1K–$3K): take profit sooner — compound toward $25K."""
    stage2_profit_target_mult: float = 1.65
    stage1_trail_activate_mult: float = 1.22
    """Stage 1: arm trail after +22% on premium.  Raised from 1.18 to keep the trail
    floor above breakeven-after-fees when trail_retrace_frac fires."""
    stop_loss_mult: float = 0.82
    """Hard stop: close when premium mult falls to this (0.82 ≈ −18% on premium paid).
    Tightened from 0.72 (−28%) — small accounts cannot absorb large per-trade losses."""
    trail_activate_mult: float = 1.25
    """Arm trailing give-back after premium reaches this multiple of entry."""
    trail_retrace_frac: float = 0.10
    """Exit trail when mult falls this fraction below session peak (after armed)."""
    min_trail_exit_mult: float = 1.08
    """Never trail-exit below this multiple of entry premium.  1.08 ensures at least
    +8% gain on exit, covering typical bid/ask spread and commissions."""
    min_dte_exit: int = 1
    """Force-exit when fewer than this many days remain to expiry.  1 day instead of 2
    gives positions one extra day of potential movement before forced close."""

    # ---- Option parameters --------------------------------------------------
    stage1_dte: int = 7
    """Days-to-expiry for Stage 1 (small account) buys — short-dated, high theta risk."""
    stage2_dte: int = 14
    """Days-to-expiry for Stage 2 buys."""
    stage3_dte: int = 21
    """Days-to-expiry for Stage 3 buys — slightly longer runway as size grows."""

    scalp_dte: int = 1
    """DTE for high-conviction intraday scalp plays. 1 = 1DTE (max gamma leverage).
    Set to 0 for true 0DTE lottery plays (extreme risk, use only with live chain data)."""
    scalp_min_alignment: float = 0.75
    scalp_min_confidence: float = 0.70
    weekly_otm_ladder: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.06)

    stage1_otm_pct: float = 0.020
    """2% OTM for Stage 1.  Raised from 1% — 1% OTM options have lottery-level win
    rates; 2% OTM still provides leverage while meaningfully improving delta and the
    probability of expiring in the money."""
    stage2_otm_pct: float = 0.010
    """1% OTM for Stage 2 — near-ATM directional."""
    stage3_otm_pct: float = 0.000
    """ATM for Stage 3 — defined-risk as account approaches PDT threshold."""

    # ---- Market parameters (fallbacks when no chain data available) ---------
    default_iv: float = 0.20
    """Fallback implied volatility (20% annualised) when no chain data is present.
    Slightly higher than historical SPY average to err on the side of conservative
    premium estimates.  Always prefer a live IV surface over this fallback."""
    assumed_daily_move_pct: float = 0.008
    """Assumed underlying daily move in a trending regime (0.8% per day)."""

    def size_fraction(self, balance: float) -> float:
        if balance < 3_000.0:
            return self.stage1_size_frac
        if balance < 10_000.0:
            return self.stage2_size_frac
        return self.stage3_size_frac

    def dte(self, balance: float) -> int:
        if balance < 3_000.0:
            return self.stage1_dte
        if balance < 10_000.0:
            return self.stage2_dte
        return self.stage3_dte

    def otm_pct(self, balance: float) -> float:
        if balance < 3_000.0:
            return self.stage1_otm_pct
        if balance < 10_000.0:
            return self.stage2_otm_pct
        return self.stage3_otm_pct

    def profit_target_for(self, balance: float) -> float:
        if balance < 3_000.0:
            return self.stage1_profit_target_mult
        if balance < 10_000.0:
            return self.stage2_profit_target_mult
        return self.stage3_profit_target_mult

    def trail_activate_for(self, balance: float) -> float:
        if balance < 3_000.0:
            return self.stage1_trail_activate_mult
        return self.trail_activate_mult


def apply_challenge_profile_env(cfg: ChallengeConfig) -> ChallengeConfig:
    """Tune challenge risk from env without changing seed/target.

    ``RLM_CHALLENGE_PROFILE=robinhood_elite`` — swing/LEAPS-leaning: longer DTE,
    smaller % of balance per entry, slightly wider OTM, wider min-DTE exit buffer.
    """
    prof = (os.environ.get("RLM_CHALLENGE_PROFILE") or "").strip().lower()
    if prof in ("", "default", "aggressive", "weekly"):
        return cfg
    if prof == "robinhood_elite":
        return replace(
            cfg,
            max_concurrent_positions=1,
            stage1_size_frac=0.12,
            stage2_size_frac=0.10,
            stage3_size_frac=0.08,
            stage1_dte=45,
            stage2_dte=60,
            stage3_dte=90,
            scalp_dte=3,
            stage1_otm_pct=0.015,
            stage2_otm_pct=0.010,
            stage3_otm_pct=0.003,
            stage3_profit_target_mult=2.2,
            stage1_profit_target_mult=1.55,
            stage2_profit_target_mult=1.75,
            stop_loss_mult=0.55,
            min_dte_exit=5,
        )
    return cfg
