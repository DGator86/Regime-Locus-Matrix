from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TransactionCostConfig:
    """
    Extra execution frictions layered on top of fill-price slippage and commissions.
    """

    extra_spread_fraction: float = 0.0
    underlying_slippage_bps: float = 0.0
    market_impact_factor: float = 0.0
    avg_daily_volume_floor: float = 1_000.0


@dataclass(frozen=True)
class TransactionCostBreakdown:
    extra_spread_cost: float
    underlying_slippage_cost: float
    market_impact_cost: float

    @property
    def total(self) -> float:
        return float(self.extra_spread_cost + self.underlying_slippage_cost + self.market_impact_cost)

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["total"] = self.total
        return payload


def calculate_transaction_cost(
    *,
    matched_legs: list[dict],
    underlying_price: float,
    quantity: int,
    contract_multiplier: int = 100,
    config: TransactionCostConfig | None = None,
) -> TransactionCostBreakdown:
    cfg = config or TransactionCostConfig()
    qty = max(int(quantity), 0)
    if qty == 0 or not matched_legs:
        return TransactionCostBreakdown(0.0, 0.0, 0.0)

    spread_cost = 0.0
    reference_premium = 0.0
    volume_ratio = 0.0

    for leg in matched_legs:
        bid = float(leg.get("bid", 0.0) or 0.0)
        ask = float(leg.get("ask", 0.0) or 0.0)
        spread = max(ask - bid, 0.0)
        mid = float(leg.get("mid", (bid + ask) / 2.0))
        volume = float(leg.get("volume", 0.0) or 0.0)

        spread_cost += spread * cfg.extra_spread_fraction * contract_multiplier * qty
        reference_premium += abs(mid) * contract_multiplier * qty

        if volume > 0.0:
            volume_ratio = max(
                volume_ratio,
                qty / max(volume, cfg.avg_daily_volume_floor),
            )

    underlying_slippage_cost = (
        abs(float(underlying_price)) * (cfg.underlying_slippage_bps / 10_000.0) * len(matched_legs) * qty
    )
    market_impact_cost = 0.0
    if cfg.market_impact_factor > 0.0 and volume_ratio > 0.0:
        market_impact_cost = reference_premium * cfg.market_impact_factor * math.sqrt(volume_ratio)

    return TransactionCostBreakdown(
        extra_spread_cost=float(spread_cost),
        underlying_slippage_cost=float(underlying_slippage_cost),
        market_impact_cost=float(market_impact_cost),
    )
