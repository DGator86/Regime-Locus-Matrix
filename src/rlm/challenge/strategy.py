"""ChallengeStrategy — translate a persona directive into a specific option play.

Selects **affordable** weeklies (5–7 DTE) on $1K–$3K books, with optional high-vol
1DTE scalps when conviction is strong.  Sizing is handled by ``ChallengeEngine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rlm.challenge.config import ChallengeConfig
from rlm.challenge.pricing import delta_per_theta_ratio, estimate_delta, estimate_premium


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
        """Return the best affordable ``PlaySpec`` or ``None``."""
        if directive == "no_trade":
            return None

        option_type: Literal["call", "put"] = "call" if directive == "long" else "put"
        direction: Literal["long", "short"] = directive  # type: ignore[assignment]

        max_pps = _max_premium_per_share(balance, cfg)
        if max_pps <= 0:
            return None

        high_vol_scalp = (
            balance < 3_000.0
            and signal_alignment >= cfg.scalp_min_alignment
            and confidence >= cfg.scalp_min_confidence
        )

        weekly_dte = max(5, int(cfg.dte(balance)))
        base_otm = cfg.otm_pct(balance)

        # Surface-aware strike selection: pick OTM% with best delta/theta ratio
        if getattr(cfg, "use_surface_strike_selection", False):
            search_range = getattr(cfg, "strike_search_otm_range", (0.000, 0.010, 0.020, 0.030, 0.040, 0.050))
            best_ratio = -1.0
            best_otm = base_otm
            for candidate_otm in search_range:
                if option_type == "call":
                    candidate_strike = underlying_price * (1.0 + candidate_otm)
                else:
                    candidate_strike = underlying_price * (1.0 - candidate_otm)
                candidate_premium = estimate_premium(underlying_price, iv, weekly_dte, candidate_strike)
                if candidate_premium > max_pps:
                    continue  # not affordable — skip
                ratio = delta_per_theta_ratio(underlying_price, candidate_strike, iv, weekly_dte, option_type)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_otm = candidate_otm
            base_otm = best_otm

        candidates: list[tuple[int, float, str]] = []

        candidates.append((weekly_dte, base_otm, f"weekly-{weekly_dte}DTE"))
        for otm in cfg.weekly_otm_ladder:
            if otm > base_otm:
                candidates.append((weekly_dte, otm, f"weekly-{weekly_dte}DTE-{otm * 100:.0f}otm"))

        if high_vol_scalp:
            candidates.insert(0, (int(cfg.scalp_dte), 0.0, f"hv-scalp-{cfg.scalp_dte}DTE-ATM"))
            candidates.insert(1, (int(cfg.scalp_dte), 0.01, f"hv-scalp-{cfg.scalp_dte}DTE-1otm"))
            tight_weekly = max(5, weekly_dte - 2)
            candidates.insert(2, (tight_weekly, base_otm, f"weekly-tight-{tight_weekly}DTE"))

        for dte, otm_pct, tag in candidates:
            spec = _build_spec(
                option_type=option_type,
                direction=direction,
                underlying_price=underlying_price,
                iv=iv,
                dte=dte,
                otm_pct=otm_pct,
                rationale=tag,
            )
            if spec.estimated_premium <= max_pps:
                return spec

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_premium_per_share(balance: float, cfg: ChallengeConfig) -> float:
    return balance * cfg.size_fraction(balance) / 100.0


def _build_spec(
    *,
    option_type: Literal["call", "put"],
    direction: Literal["long", "short"],
    underlying_price: float,
    iv: float,
    dte: int,
    otm_pct: float,
    rationale: str,
) -> PlaySpec:
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


def _round_strike(price: float) -> float:
    """Round to nearest dollar — standard equity option increment."""
    return round(price)
