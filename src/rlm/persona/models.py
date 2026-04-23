"""Typed models for the four-stage persona pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Pipeline input — consumed from existing RLM pipeline outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonaPipelineInput:
    """Normalised inputs consumed from existing RLM outputs.

    All values should be pre-computed by the caller from standard RLM
    pipeline results.  No computation happens here.
    """
    symbol: str

    # From regime / forecast outputs
    regime_label: str = "unknown"          # e.g. "bull_trend", "bear_vol"
    regime_confidence: float = 0.5        # 0..1
    forecast_return: float = 0.0          # expected return, signed
    realized_vol: float = 0.2            # annualised σ

    # From factor pipeline
    signal_alignment: float = 0.5        # 0..1 — how many factors agree
    momentum_score: float = 0.0          # signed, normalised
    mean_reversion_score: float = 0.0    # signed, normalised

    # From microstructure / options flow (optional)
    dealer_gamma_exposure: float = 0.0   # positive = supportive
    options_put_call_ratio: float = 1.0  # >1 bearish skew
    bid_ask_spread_pct: float = 0.01     # 0..1
    volume_ratio: float = 1.0            # today vs avg

    # Historical edge (from Data stage / backtest ledger)
    historical_edge: float = 0.5         # 0..1 win-rate proxy for this regime


# ---------------------------------------------------------------------------
# Stage outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SevenStageOutput:
    """Signal normalisation and structured interpretation."""
    bias: Literal["bullish", "bearish", "neutral"]
    signal_alignment: float          # 0..1
    confidence: float                # 0..1
    regime_label: str
    notes: str = ""


@dataclass(frozen=True)
class GarakStageOutput:
    """Trap / deception / false-breakout detection."""
    trap_risk: float                 # 0..1  (higher = more suspicious)
    dealer_alignment: Literal["supportive", "neutral", "opposed"]
    liquidity_comment: str
    veto: bool                       # True → block trade regardless of Seven


@dataclass(frozen=True)
class SiskoStageOutput:
    """Final trade directive authority."""
    directive: Literal["long", "short", "no_trade"]
    entry_policy: str
    invalidation_policy: str
    target_policy: str
    reason: str = ""


@dataclass(frozen=True)
class DataStageOutput:
    """Post-trade audit / empirical validation / edge tracking."""
    regime_match: Literal["high", "moderate", "low"]
    historical_edge: float           # 0..1
    adaptation_note: str
    review_flag: bool


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonaPipelineResult:
    """Complete persona pipeline result."""
    symbol: str
    seven: SevenStageOutput
    garak: GarakStageOutput
    sisko: SiskoStageOutput
    data: DataStageOutput

    def to_dict(self) -> dict:  # type: ignore[return]
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "symbol": self.symbol,
            "seven": {
                "bias": self.seven.bias,
                "signal_alignment": self.seven.signal_alignment,
                "confidence": self.seven.confidence,
                "regime_label": self.seven.regime_label,
                "notes": self.seven.notes,
            },
            "garak": {
                "trap_risk": self.garak.trap_risk,
                "dealer_alignment": self.garak.dealer_alignment,
                "liquidity_comment": self.garak.liquidity_comment,
                "veto": self.garak.veto,
            },
            "sisko": {
                "directive": self.sisko.directive,
                "entry_policy": self.sisko.entry_policy,
                "invalidation_policy": self.sisko.invalidation_policy,
                "target_policy": self.sisko.target_policy,
                "reason": self.sisko.reason,
            },
            "data": {
                "regime_match": self.data.regime_match,
                "historical_edge": self.data.historical_edge,
                "adaptation_note": self.data.adaptation_note,
                "review_flag": self.data.review_flag,
            },
        }
