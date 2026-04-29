"""ChallengeConfig — parameters for the $1K→$25K dry-run options challenge."""

from __future__ import annotations

from dataclasses import dataclass


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
    max_concurrent_positions: int = 2
    """Never hold more than this many open option positions simultaneously."""

    # ---- Sizing by account stage (fraction of balance per trade) ------------
    stage1_size_frac: float = 0.25
    """$1K – $3K: 25% of balance in premium per trade."""
    stage2_size_frac: float = 0.20
    """$3K – $10K: 20% of balance in premium per trade."""
    stage3_size_frac: float = 0.15
    """$10K – $25K: 15% of balance in premium per trade."""

    # ---- Exit rules ---------------------------------------------------------
    profit_target_mult: float = 2.0
    """Close position when option value reaches this multiple of entry premium."""
    stop_loss_mult: float = 0.50
    """Close position when option value falls to this fraction of entry premium."""
    min_dte_exit: int = 2
    """Force-exit when fewer than this many days remain to expiry."""

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

    stage1_otm_pct: float = 0.010
    """1% OTM for Stage 1 — lottery-style leverage."""
    stage2_otm_pct: float = 0.005
    """0.5% OTM for Stage 2 — near-ATM directional."""
    stage3_otm_pct: float = 0.000
    """ATM for Stage 3 — defined-risk as account approaches PDT threshold."""

    # ---- Market parameters (fallbacks when no chain data available) ---------
    default_iv: float = 0.18
    """Fallback implied volatility (18% annualised) when no chain data is present."""
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
