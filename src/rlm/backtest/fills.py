from __future__ import annotations

from dataclasses import dataclass, field

from rlm.backtest.slippage import SlippageConfig, compute_leg_slippage


@dataclass(frozen=True)
class FillConfig:
    contract_multiplier: int = 100
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    liquidity_impact_factor: float = 0.5
    min_size_floor: float = 1.0


def entry_fill_price(
    *,
    side: str,
    bid: float,
    ask: float,
    config: FillConfig | None = None,
    realized_vol: float | None = None,
    quantity: int = 1,
    quote_size: float | None = None,
) -> float:
    """
    Returns option premium per contract for entry.
    Long pays ask + slippage.
    Short receives bid - slippage.

    ``realized_vol`` enables volatility-scaled slippage when provided.
    """
    cfg = config or FillConfig()
    slip = compute_leg_slippage(bid=bid, ask=ask, config=cfg.slippage, realized_vol=realized_vol)

    impact = 0.0
    if quote_size is not None:
        depth = max(float(quote_size), cfg.min_size_floor)
        impact = slip * cfg.liquidity_impact_factor * max(float(quantity) / depth - 1.0, 0.0)

    if side == "long":
        return max(ask + slip + impact, 0.0)
    if side == "short":
        return max(bid - slip - impact, 0.0)

    raise ValueError(f"Unknown side: {side}")


def exit_fill_price(
    *,
    side: str,
    bid: float,
    ask: float,
    config: FillConfig | None = None,
    realized_vol: float | None = None,
    quantity: int = 1,
    quote_size: float | None = None,
) -> float:
    """
    Returns option premium per contract for exit.
    Long exits by selling at bid - slippage.
    Short exits by buying back at ask + slippage.

    ``realized_vol`` enables volatility-scaled slippage when provided.
    """
    cfg = config or FillConfig()
    slip = compute_leg_slippage(bid=bid, ask=ask, config=cfg.slippage, realized_vol=realized_vol)

    impact = 0.0
    if quote_size is not None:
        depth = max(float(quote_size), cfg.min_size_floor)
        impact = slip * cfg.liquidity_impact_factor * max(float(quantity) / depth - 1.0, 0.0)

    if side == "long":
        return max(bid - slip - impact, 0.0)
    if side == "short":
        return max(ask + slip + impact, 0.0)

    raise ValueError(f"Unknown side: {side}")


def signed_cashflow_for_fill(
    *,
    side: str,
    premium: float,
    contract_multiplier: int = 100,
) -> float:
    """
    Positive means cash outflow from account.
    Long = pay premium.
    Short = receive premium (negative outflow).
    """
    if side == "long":
        return premium * contract_multiplier
    if side == "short":
        return -premium * contract_multiplier
    raise ValueError(f"Unknown side: {side}")
