"""AggressiveSizer — position sizing for rapid small-account growth.

Sizing is stage-aware:
  Stage 1 ($1K–$3K)  : 25% of balance in premium → maximum aggression
  Stage 2 ($3K–$10K) : 20% of balance in premium → controlled aggression
  Stage 3 ($10K–$25K): 15% of balance in premium → momentum with discipline
"""

from __future__ import annotations

from rlm.challenge.config import ChallengeConfig


class AggressiveSizer:
    """Determine contract quantity and premium spend for a new position."""

    def compute(
        self,
        balance: float,
        premium_per_share: float,
        cfg: ChallengeConfig,
    ) -> tuple[int, float]:
        """Return ``(qty_contracts, actual_spend)``.

        *qty_contracts* is at least 1 if balance covers the premium.
        *actual_spend* is the true cash committed (may be less than max_spend
        when contract rounding leaves a remainder).

        Returns ``(0, 0.0)`` when the account cannot afford even one contract.
        """
        frac = cfg.size_fraction(balance)
        max_spend = balance * frac
        cost_per_contract = premium_per_share * 100.0  # 1 contract = 100 shares

        if cost_per_contract <= 0 or cost_per_contract > balance:
            return 0, 0.0

        qty = max(1, int(max_spend / cost_per_contract))
        actual_spend = qty * cost_per_contract

        # Never spend more than the full balance
        if actual_spend > balance:
            qty = max(0, qty - 1)
            actual_spend = qty * cost_per_contract

        return qty, actual_spend
