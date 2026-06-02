"""ChallengeConfig — parameters for the $1K→$100K options challenge.

Compliant with FINRA Rule 4210 (amended), effective June 4, 2026.
The Pattern Day Trader rule and $25,000 minimum equity requirement have been
eliminated.  Day trading is now unrestricted for cash accounts and margin
accounts with ≥$2,000 equity.  The challenge uses a cash account (long options,
fully paid) — no intraday margin constraint applies beyond not exceeding balance.
"""

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
    ChallengeMilestone(5_000.0,   "Stage I — Ignition",    "5x from seed; prove the edge"),
    ChallengeMilestone(20_000.0,  "Stage II — Momentum",   "4x from Stage I; compound the gains"),
    ChallengeMilestone(50_000.0,  "Stage III — Scale",     "2.5x from Stage II; disciplined sizing"),
    ChallengeMilestone(100_000.0, "Stage IV — Arrival",    "2x from Stage III; $100K target reached"),
)


COMPLIANCE_NOTE = (
    "This simulator is designed for compliance with FINRA Rule 4210 (amended), "
    "effective June 4, 2026 (SR-FINRA-2025-017). The Pattern Day Trader designation "
    "and $25,000 minimum equity requirement have been eliminated. "
    "Unlimited day trades are permitted. Cash accounts (long options, fully paid) "
    "have no intraday margin requirement. Margin accounts require $2,000 minimum "
    "and are subject to real-time intraday exposure monitoring."
)


@dataclass(frozen=True)
class ChallengeConfig:
    """Full configuration for one challenge run.

    Defaults are tuned for an aggressive small-account growth strategy.
    Override only the knobs you need.
    """

    # ---- Account parameters -------------------------------------------------
    seed_capital: float = 1_000.0
    target_capital: float = 100_000.0
    challenge_days: int = 252
    """Trading days in the challenge window (12 months)."""
    symbol: str = "SPY"

    # ---- Position management ------------------------------------------------
    max_concurrent_positions: int = 2

    # ---- Sizing by account stage (fraction of balance per trade) ------------
    stage1_size_frac: float = 0.35
    """$1K–$5K: up to 35% of balance in premium per entry."""
    stage2_size_frac: float = 0.25
    """$5K–$20K: 25% of balance in premium per trade."""
    stage3_size_frac: float = 0.18
    """$20K–$50K: 18% of balance in premium per trade."""
    stage4_size_frac: float = 0.12
    """$50K–$100K: 12% of balance in premium per trade."""

    # ---- Exit rules ---------------------------------------------------------
    stage1_profit_target_mult: float = 1.50
    """Stage 1 ($1K–$5K): take profit at +50% on premium."""
    stage2_profit_target_mult: float = 1.75
    """Stage 2 ($5K–$20K): take profit at +75% on premium."""
    stage3_profit_target_mult: float = 2.00
    """Stage 3 ($20K–$50K): take profit at +100% on premium."""
    stage4_profit_target_mult: float = 1.75
    """Stage 4 ($50K–$100K): +75% (conservative at scale)."""

    stage1_trail_activate_mult: float = 1.22
    """Stage 1: arm trail after +22% on premium."""
    stage2_trail_activate_mult: float = 1.30
    """Stage 2: arm trail after +30% on premium."""
    stage3_trail_activate_mult: float = 1.40
    """Stage 3: arm trail after +40% on premium."""
    stage4_trail_activate_mult: float = 1.30
    """Stage 4: arm trail after +30% on premium."""

    # Per-stage stop losses (tighten as account grows)
    stage1_stop_loss_mult: float = 0.80
    """Stage 1 hard stop: -20% on premium."""
    stage2_stop_loss_mult: float = 0.82
    """Stage 2 hard stop: -18% on premium."""
    stage3_stop_loss_mult: float = 0.85
    """Stage 3 hard stop: -15% on premium."""
    stage4_stop_loss_mult: float = 0.88
    """Stage 4 hard stop: -12% on premium."""

    # Fallback trail fields
    trail_activate_mult: float = 1.25
    """Fallback trail activation multiple (used if stage not matched)."""
    trail_retrace_frac: float = 0.10
    """Exit trail when mult falls this fraction below session peak (after armed)."""
    min_trail_exit_mult: float = 1.08
    """Never trail-exit below this multiple of entry premium."""
    min_dte_exit: int = 1
    """Force-exit when fewer than this many days remain to expiry."""

    # ---- Option parameters --------------------------------------------------
    stage1_dte: int = 7
    """Days-to-expiry for Stage 1 buys."""
    stage2_dte: int = 14
    """Days-to-expiry for Stage 2 buys."""
    stage3_dte: int = 21
    """Days-to-expiry for Stage 3 buys."""
    stage4_dte: int = 21
    """Days-to-expiry for Stage 4 buys."""

    scalp_dte: int = 1
    """DTE for high-conviction intraday scalp plays."""
    scalp_min_alignment: float = 0.75
    scalp_min_confidence: float = 0.70
    weekly_otm_ladder: tuple[float, ...] = (0.010, 0.015, 0.020, 0.030, 0.040)

    stage1_otm_pct: float = 0.015
    """1.5% OTM for Stage 1."""
    stage2_otm_pct: float = 0.010
    """1.0% OTM for Stage 2 — near-ATM directional."""
    stage3_otm_pct: float = 0.005
    """0.5% OTM for Stage 3."""
    stage4_otm_pct: float = 0.000
    """ATM for Stage 4."""

    # ---- Market parameters (fallbacks when no chain data available) ---------
    default_iv: float = 0.20
    """Fallback implied volatility (20% annualised)."""
    assumed_daily_move_pct: float = 0.008
    """Assumed underlying daily move in a trending regime (0.8% per day)."""

    # ---- Friction / execution cost model ------------------------------------
    spread_half_width_frac: float = 0.08
    """Half-spread as a fraction of premium (8%)."""
    commission_per_contract: float = 0.65
    """Per-leg, per-contract commission in dollars."""
    use_spread_model: bool = True
    """When True, apply spread and commission friction to every entry and exit."""

    # ---- IV proxy parameters ------------------------------------------------
    iv_vol_premium: float = 0.15
    """Fractional markup applied to realized_vol when used as an IV proxy."""

    # ---- Event calendar gate ------------------------------------------------
    block_hours_before_event: int = 24
    """No new entries within this many hours of a known macro event."""
    major_event_dates: tuple[str, ...] = ()
    """ISO date strings (YYYY-MM-DD) of major events: FOMC, CPI, NFP, etc."""

    # ---- Win-rate filter per regime key -------------------------------------
    regime_win_rate_min: float = 0.40
    """Minimum rolling win-rate required to enter a trade in a given regime."""
    regime_win_rate_min_samples: int = 5
    """Only apply the win-rate gate once this many samples exist for a regime."""

    # ---- Correlation / basket risk gate -------------------------------------
    max_same_direction_premium_frac: float = 0.50
    """Max fraction of balance committed to same-direction positions before blocking entry."""

    # ---- Per-stage daily loss limit -----------------------------------------
    stage1_max_daily_loss_frac: float = 0.075
    """Halt Stage 1 entries if daily realized loss exceeds 7.5% of balance."""
    stage2_max_daily_loss_frac: float = 0.050
    """Halt Stage 2 entries if daily realized loss exceeds 5% of balance."""
    stage3_max_daily_loss_frac: float = 0.035
    """Halt Stage 3 entries if daily realized loss exceeds 3.5% of balance."""
    stage4_max_daily_loss_frac: float = 0.025
    """Halt Stage 4 entries if daily realized loss exceeds 2.5% of balance (most conservative)."""

    # ---- Theta/IV surface strike selection ----------------------------------
    use_surface_strike_selection: bool = True
    """When True, pick strike with highest delta/theta ratio from the OTM range."""
    strike_search_otm_range: tuple[float, ...] = (0.000, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050)
    """OTM fractions to evaluate when use_surface_strike_selection is True."""

    # ---- Pace tracking ------------------------------------------------------
    pace_boost_max: float = 0.20
    """Max fractional sizing boost when behind pace."""
    pace_reduce_max: float = 0.15
    """Max fractional sizing reduction when ahead of pace."""
    pace_boost_threshold: float = 0.80
    """Boost kicks in when progress_ratio < this."""
    pace_reduce_threshold: float = 1.30
    """Reduce kicks in when progress_ratio > this."""

    # ---- FINRA Rule 4210 compliance (effective June 4, 2026) ----------------
    account_type: str = "cash"
    """Account type: 'cash' (options buying, no margin) or 'margin' (margin account, $2K min).
    Cash accounts are fully FINRA-compliant for buying long options with any balance.
    Margin accounts require minimum $2,000 and are subject to intraday exposure monitoring."""

    margin_account_minimum: float = 2_000.0
    """FINRA minimum equity for margin accounts under amended Rule 4210.
    Not applicable to cash accounts."""

    intraday_exposure_limit_frac: float = 1.0
    """Maximum total open option premium exposure as a fraction of current balance.
    1.0 = exposure limited to full account equity (cash account behavior — always compliant).
    For margin accounts, brokers may allow > 1.0 based on real-time margin excess."""

    end_of_day_margin_call_enabled: bool = False
    """When True, simulate end-of-day margin call: force-reduce positions if
    total open exposure exceeds balance × intraday_exposure_limit_frac at session end.
    Mirrors the broker 'Path B' (end-of-day calculation) compliance option."""

    # ---- Stage-aware helpers ------------------------------------------------

    def size_fraction(self, balance: float) -> float:
        if balance < 5_000.0:
            return self.stage1_size_frac
        if balance < 20_000.0:
            return self.stage2_size_frac
        if balance < 50_000.0:
            return self.stage3_size_frac
        return self.stage4_size_frac

    def dte(self, balance: float) -> int:
        if balance < 5_000.0:
            return self.stage1_dte
        if balance < 20_000.0:
            return self.stage2_dte
        if balance < 50_000.0:
            return self.stage3_dte
        return self.stage4_dte

    def otm_pct(self, balance: float) -> float:
        if balance < 5_000.0:
            return self.stage1_otm_pct
        if balance < 20_000.0:
            return self.stage2_otm_pct
        if balance < 50_000.0:
            return self.stage3_otm_pct
        return self.stage4_otm_pct

    def profit_target_for(self, balance: float) -> float:
        if balance < 5_000.0:
            return self.stage1_profit_target_mult
        if balance < 20_000.0:
            return self.stage2_profit_target_mult
        if balance < 50_000.0:
            return self.stage3_profit_target_mult
        return self.stage4_profit_target_mult

    def trail_activate_for(self, balance: float) -> float:
        if balance < 5_000.0:
            return self.stage1_trail_activate_mult
        if balance < 20_000.0:
            return self.stage2_trail_activate_mult
        if balance < 50_000.0:
            return self.stage3_trail_activate_mult
        return self.stage4_trail_activate_mult

    def stop_loss_for(self, balance: float) -> float:
        """Per-stage stop loss multiple (tightens as account grows)."""
        if balance < 5_000.0:
            return self.stage1_stop_loss_mult
        if balance < 20_000.0:
            return self.stage2_stop_loss_mult
        if balance < 50_000.0:
            return self.stage3_stop_loss_mult
        return self.stage4_stop_loss_mult

    def max_daily_loss_frac(self, balance: float) -> float:
        if balance < 5_000.0:
            return self.stage1_max_daily_loss_frac
        if balance < 20_000.0:
            return self.stage2_max_daily_loss_frac
        if balance < 50_000.0:
            return self.stage3_max_daily_loss_frac
        return self.stage4_max_daily_loss_frac


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
            stage4_size_frac=0.06,
            stage1_dte=45,
            stage2_dte=60,
            stage3_dte=90,
            stage4_dte=90,
            scalp_dte=3,
            stage1_otm_pct=0.015,
            stage2_otm_pct=0.010,
            stage3_otm_pct=0.003,
            stage4_otm_pct=0.000,
            stage1_profit_target_mult=1.55,
            stage2_profit_target_mult=1.75,
            stage3_profit_target_mult=2.20,
            stage4_profit_target_mult=1.90,
            stage1_stop_loss_mult=0.55,
            stage2_stop_loss_mult=0.60,
            stage3_stop_loss_mult=0.65,
            stage4_stop_loss_mult=0.70,
            min_dte_exit=5,
        )
    return cfg
