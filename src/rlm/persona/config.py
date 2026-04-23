"""PersonaConfig — thresholds for the four-stage persona decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    """Tunable thresholds for all four persona stages.

    All defaults are conservative; override only what you need.
    """

    # Seven — signal interpretation
    signal_alignment_threshold: float = 0.55
    """Minimum signal_alignment score to allow a directional directive."""
    confidence_threshold: float = 0.45
    """Minimum confidence score to allow a directional directive."""
    direction_score_threshold: float = 0.2
    """Minimum |S_D| to treat the signal as directionally meaningful."""

    # Garak — trap / deception detection
    trap_risk_veto_threshold: float = 0.65
    """trap_risk at or above this value triggers a Garak veto."""
    vol_risk_weight: float = 0.40
    """Weight of volatility stress in the trap_risk composite."""
    liq_risk_weight: float = 0.40
    """Weight of liquidity stress in the trap_risk composite."""
    dealer_risk_weight: float = 0.20
    """Weight of opposed dealer flow in the trap_risk composite."""

    # Sisko — directive authority
    min_confidence_for_directional: float = 0.40
    """Confidence below this always produces no_trade."""

    # Data — audit / edge tracking
    regime_match_high_threshold: float = 0.65
    """historical_edge at or above this → regime_match "high"."""
    regime_match_moderate_threshold: float = 0.40
    """historical_edge at or above this → regime_match "moderate"."""
    review_flag_edge_threshold: float = 0.42
    """historical_edge below this sets review_flag=True."""
