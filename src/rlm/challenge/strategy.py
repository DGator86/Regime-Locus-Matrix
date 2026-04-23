"""ChallengeStrategy — translate a persona directive into a specific option play.

This module selects *what* to buy (call vs put, OTM level, DTE) based on
the persona pipeline directive and the current account stage.  It does not
execute or size the trade — those responsibilities belong to ChallengeEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.pricing import estimate_delta, estimate_premium


@dataclass(frozen=True)
class PlaySpec:
    """Parameters for a specific option play selected by ChallengeStrategy."""

    option_type: Literal["call", "put"]
    direction: Literal["long", "short"]
    strike: float
    dte: int
    estimated_premium: float
    estimated_delta: float
    otm_pct: float
    rationale: str


class ChallengeStrategy:
    """Select an aggressive option play given a persona directive."""

    def select(
        self,
        directive: Literal["long", "short", "no_trade"],
        underlying_price: float,
        balance: float,
        iv: float,
        cfg: ChallengeConfig,
        signal_alignment: float = 0.7,
        confidence: float = 0.7,
    ) -> PlaySpec | None:
        """Return a ``PlaySpec`` or ``None`` (no trade).

        Extra-high confidence/alignment unlocks tighter DTE (more leverage);
        borderline signals get standard DTE.
        """
        if directive == "no_trade":
            return None

        option_type: Literal["call", "put"] = "call" if directive == "long" else "put"
        direction: Literal["long", "short"] = directive  # type: ignore[assignment]

        base_dte = cfg.dte(balance)
        base_otm = cfg.otm_pct(balance)

        # Ultra-high conviction: compress DTE for extra leverage (Stage 1 only)
        if balance < 3_000 and signal_alignment >= 0.80 and confidence >= 0.75:
            dte = max(3, base_dte - 3)
            otm_pct = base_otm + 0.005  # push slightly further OTM for more delta leverage
            rationale = "high-conviction scalp: compressed DTE + extended OTM for max leverage"
        else:
            dte = base_dte
            otm_pct = base_otm
            rationale = f"stage {_stage_label(balance, cfg)} directional: {option_type} {dte}DTE"

        # Compute strike
        if option_type == "call":
            strike = _round_strike(underlying_price * (1.0 + otm_pct))
        else:
            strike = _round_strike(underlying_price * (1.0 - otm_pct))

        premium = estimate_premium(underlying_price, iv, dte, strike)
        delta = estimate_delta(underlying_price, strike, iv, dte, option_type)

        return PlaySpec(
            option_type=option_type,
            direction=direction,
            strike=strike,
            dte=dte,
            estimated_premium=premium,
            estimated_delta=delta,
            otm_pct=otm_pct,
            rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_strike(price: float) -> float:
    """Round to nearest dollar — standard equity option increment."""
    return round(price)


def _stage_label(balance: float, cfg: ChallengeConfig) -> str:
    if balance < 3_000:
        return "1"
    if balance < 10_000:
        return "2"
    return "3"
