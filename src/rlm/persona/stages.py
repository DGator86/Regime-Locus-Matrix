"""Four deterministic stage functions for the persona decision pipeline.

Seven  → signal normalization + bias determination
Garak  → trap / deception / false-breakout assessment
Sisko  → final trade directive
Data   → empirical audit and edge validation
"""

from __future__ import annotations

from typing import Literal

from rlm.persona.config import PersonaConfig
from rlm.persona.models import (
    DataStageOutput,
    GarakStageOutput,
    PersonaInputs,
    SevenStageOutput,
    SiskoStageOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _norm_score(score: float) -> float:
    """Map a score from [-1, 1] into [0, 1] (higher = more positive direction)."""
    return _clamp((score + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Stage 1 — Seven
# ---------------------------------------------------------------------------

def run_seven(inputs: PersonaInputs, cfg: PersonaConfig) -> SevenStageOutput:
    """Normalise raw factor scores into a structured bias/alignment/confidence triple."""

    regime = inputs.direction_regime.lower()

    # Determine directional bias from regime label, with S_D as tie-breaker
    if regime == "bull" or (regime == "neutral" and inputs.s_d > cfg.direction_score_threshold):
        bias: Literal["bullish", "bearish", "neutral"] = "bullish"
        sign = 1.0
    elif regime == "bear" or (regime == "neutral" and inputs.s_d < -cfg.direction_score_threshold):
        bias = "bearish"
        sign = -1.0
    else:
        bias = "neutral"
        sign = 0.0

    # Signal alignment: how well do direction, liquidity, and dealer scores point
    # in the same direction as the stated bias.  Weights: S_D=0.5, S_L=0.3, S_G=0.2
    if sign != 0.0:
        aligned_d = _clamp(inputs.s_d * sign)
        aligned_l = _clamp(inputs.s_l * sign)
        aligned_g = _clamp(inputs.s_g * sign)
        signal_alignment = _clamp(0.5 * aligned_d + 0.3 * aligned_l + 0.2 * aligned_g)
    else:
        signal_alignment = 0.0

    # Confidence: blends HMM state certainty with raw direction-score magnitude
    signal_strength = _clamp(abs(inputs.s_d))
    confidence = _clamp(0.6 * inputs.hmm_confidence + 0.4 * signal_strength)

    return SevenStageOutput(
        bias=bias,
        signal_alignment=signal_alignment,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Garak
# ---------------------------------------------------------------------------

def run_garak(
    inputs: PersonaInputs,
    seven: SevenStageOutput,
    cfg: PersonaConfig,
) -> GarakStageOutput:
    """Assess trap risk, dealer alignment, and liquidity quality."""

    # Volatility stress: high S_V → elevated vol → trappy conditions
    vol_risk = _norm_score(inputs.s_v)

    # Liquidity stress: low S_L → thin book → false breakouts more likely
    liq_risk = _norm_score(-inputs.s_l)

    # Dealer flow opposition: binary flag weighted by cfg
    dealer_risk = 1.0 if inputs.dealer_flow_regime.lower() == "opposed" else 0.0

    trap_risk = _clamp(
        cfg.vol_risk_weight * vol_risk
        + cfg.liq_risk_weight * liq_risk
        + cfg.dealer_risk_weight * dealer_risk
    )

    # Dealer alignment label
    dr = inputs.dealer_flow_regime.lower()
    if dr == "supportive":
        dealer_alignment: Literal["supportive", "neutral", "opposed"] = "supportive"
    elif dr == "opposed":
        dealer_alignment = "opposed"
    else:
        dealer_alignment = "neutral"

    # Liquidity comment
    if inputs.s_l > 0.5:
        liquidity_comment = "breakout appears clean with supportive liquidity"
    elif inputs.s_l > 0.1:
        liquidity_comment = "liquidity adequate but not exceptional"
    elif inputs.s_l > -0.3:
        liquidity_comment = "liquidity somewhat thin; monitor for absorption"
    else:
        liquidity_comment = "liquidity stressed; risk of false breakout elevated"

    veto = trap_risk >= cfg.trap_risk_veto_threshold

    return GarakStageOutput(
        trap_risk=trap_risk,
        dealer_alignment=dealer_alignment,
        liquidity_comment=liquidity_comment,
        veto=veto,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Sisko
# ---------------------------------------------------------------------------

def run_sisko(
    inputs: PersonaInputs,
    seven: SevenStageOutput,
    garak: GarakStageOutput,
    cfg: PersonaConfig,
) -> SiskoStageOutput:
    """Synthesise Seven + Garak into a final trade directive and execution policy."""

    # Determine directive — most restrictive gate wins
    if garak.veto:
        directive: Literal["long", "short", "no_trade"] = "no_trade"
    elif seven.confidence < cfg.min_confidence_for_directional:
        directive = "no_trade"
    elif seven.signal_alignment < cfg.signal_alignment_threshold:
        directive = "no_trade"
    elif seven.bias == "bullish":
        directive = "long"
    elif seven.bias == "bearish":
        directive = "short"
    else:
        directive = "no_trade"

    # Execution policies
    if directive == "long":
        entry_policy = "take breakout continuation on confirmed momentum"
        invalidation_policy = "fail back below trigger zone invalidates setup"
        target_policy = "scale at first expansion; trail remainder"
    elif directive == "short":
        entry_policy = "enter on rejection from overhead supply zone"
        invalidation_policy = "reclaim above trigger cancels bearish thesis"
        target_policy = "take partial at first support; trail to cost basis"
    else:
        entry_policy = "stand aside; wait for cleaner setup"
        invalidation_policy = "n/a"
        target_policy = "n/a"

    return SiskoStageOutput(
        directive=directive,
        entry_policy=entry_policy,
        invalidation_policy=invalidation_policy,
        target_policy=target_policy,
    )


# ---------------------------------------------------------------------------
# Stage 4 — Data
# ---------------------------------------------------------------------------

def run_data(
    inputs: PersonaInputs,
    sisko: SiskoStageOutput,
    cfg: PersonaConfig,
) -> DataStageOutput:
    """Emit empirical audit context and historical-edge estimate."""

    # Historical edge: use backtest metrics when available, else estimate
    if inputs.backtest_metrics:
        sharpe = float(inputs.backtest_metrics.get("sharpe_ratio") or 0.0)
        win_rate = float(inputs.backtest_metrics.get("win_rate") or 0.5)
        # Blend normalised Sharpe (cap at 2.0) and win_rate
        historical_edge = _clamp(0.5 * _clamp(sharpe / 2.0) + 0.5 * win_rate)
    else:
        # Conservative estimate from factor score magnitudes
        mag = _clamp(
            0.5 * abs(inputs.s_d)
            + 0.3 * abs(inputs.s_l)
            + 0.2 * abs(inputs.s_g)
        )
        # Anchor near 0.5; deviate proportionally to factor conviction
        historical_edge = _clamp(0.35 + 0.30 * mag)

    # Regime match quality
    if historical_edge >= cfg.regime_match_high_threshold:
        regime_match: Literal["high", "moderate", "low"] = "high"
    elif historical_edge >= cfg.regime_match_moderate_threshold:
        regime_match = "moderate"
    else:
        regime_match = "low"

    # Adaptation note keyed on volatility regime
    vr = inputs.volatility_regime.lower()
    if "high" in vr:
        adaptation_note = (
            "similar setups show reduced edge in elevated vol; size conservatively"
        )
    elif "low" in vr:
        adaptation_note = (
            "similar setups perform best in expanding vol; watch for vol expansion trigger"
        )
    else:
        adaptation_note = "similar setups perform in line with historical averages"

    review_flag = historical_edge < cfg.review_flag_edge_threshold

    return DataStageOutput(
        regime_match=regime_match,
        historical_edge=historical_edge,
        adaptation_note=adaptation_note,
        review_flag=review_flag,
    )
