from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageConfig:
    per_contract_flat: float = 0.0
    spread_fraction: float = 0.25
    min_slippage: float = 0.0


def compute_leg_slippage(
    *,
    bid: float,
    ask: float,
    config: SlippageConfig | None = None,
) -> float:
    """
    Returns slippage in option-price units, per contract, per leg.
    """
    cfg = config or SlippageConfig()
    spread = max(ask - bid, 0.0)
    slip = max(cfg.min_slippage, spread * cfg.spread_fraction)
    return slip + cfg.per_contract_flat
