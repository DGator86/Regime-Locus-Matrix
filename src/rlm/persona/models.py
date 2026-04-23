"""Typed output models for the four persona stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class PersonaInputs:
    """Scalar inputs extracted from the last bar of a PipelineResult."""

    # Composite factor scores from factors_df — all nominally in [-1, 1]
    s_d: float
    """Direction composite score (S_D)."""
    s_v: float
    """Volatility composite score (S_V)."""
    s_l: float
    """Liquidity composite score (S_L)."""
    s_g: float
    """GEX / dealer-flow composite score (S_G)."""

    # Regime classification from policy_df
    direction_regime: str
    """'bull' | 'bear' | 'neutral' from classify_state_matrix."""
    volatility_regime: str
    """'low_vol' | 'high_vol' | 'neutral'."""
    liquidity_regime: str
    """'high_liquidity' | 'low_liquidity' | 'neutral'."""
    dealer_flow_regime: str
    """'supportive' | 'opposed' | 'neutral'."""

    # Forecast confidence from forecast_df
    hmm_confidence: float
    """Max HMM state probability (0–1); falls back to 0.5 if unavailable."""

    # ROEE output
    roee_action: str | None
    """Last ROEE action string, e.g. 'enter' | 'skip' | 'hold'."""

    # Optional historical context
    backtest_metrics: dict[str, float] | None = None
    """Backtest summary dict from PipelineResult, if available."""


@dataclass(frozen=True)
class SevenStageOutput:
    """Signal normalization and structured interpretation."""

    bias: Literal["bullish", "bearish", "neutral"]
    signal_alignment: float
    """Fraction of direction/liquidity/dealer scores aligned with bias (0–1)."""
    confidence: float
    """Blended HMM-confidence + signal-strength measure (0–1)."""


@dataclass(frozen=True)
class GarakStageOutput:
    """Trap / deception / false-breakout detection."""

    trap_risk: float
    """Composite trap-risk score (0–1); higher = more suspicious."""
    dealer_alignment: Literal["supportive", "neutral", "opposed"]
    liquidity_comment: str
    """Human-readable assessment of current liquidity conditions."""
    veto: bool
    """True when trap_risk exceeds the configured veto threshold."""


@dataclass(frozen=True)
class SiskoStageOutput:
    """Final trade directive authority."""

    directive: Literal["long", "short", "no_trade"]
    entry_policy: str
    invalidation_policy: str
    target_policy: str


@dataclass(frozen=True)
class DataStageOutput:
    """Post-trade audit, empirical validation, and edge tracking."""

    regime_match: Literal["high", "moderate", "low"]
    """Quality of current regime vs. historically-edge-positive setups."""
    historical_edge: float
    """Estimated win-rate proxy or backtest-derived edge (0–1)."""
    adaptation_note: str
    """Contextual note on how similar setups have performed."""
    review_flag: bool
    """True when historical_edge is below the review threshold."""


@dataclass(frozen=True)
class PersonaPipelineResult:
    """Bundled output from all four persona stages."""

    seven: SevenStageOutput
    garak: GarakStageOutput
    sisko: SiskoStageOutput
    data: DataStageOutput

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "seven": {
                "bias": self.seven.bias,
                "signal_alignment": round(self.seven.signal_alignment, 4),
                "confidence": round(self.seven.confidence, 4),
            },
            "garak": {
                "trap_risk": round(self.garak.trap_risk, 4),
                "dealer_alignment": self.garak.dealer_alignment,
                "liquidity_comment": self.garak.liquidity_comment,
                "veto": self.garak.veto,
            },
            "sisko": {
                "directive": self.sisko.directive,
                "entry_policy": self.sisko.entry_policy,
                "invalidation_policy": self.sisko.invalidation_policy,
                "target_policy": self.sisko.target_policy,
            },
            "data": {
                "regime_match": self.data.regime_match,
                "historical_edge": round(self.data.historical_edge, 4),
                "adaptation_note": self.data.adaptation_note,
                "review_flag": self.data.review_flag,
            },
        }
