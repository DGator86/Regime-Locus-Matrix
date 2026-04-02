from __future__ import annotations

from dataclasses import dataclass

from rlm.backtest.slippage import SlippageConfig, compute_leg_slippage


@dataclass(frozen=True)
class FillConfig:
    contract_multiplier: int = 100
    slippage: SlippageConfig = SlippageConfig()


def entry_fill_price(
    *,
    side: str,
    bid: float,
    ask: float,
    config: FillConfig | None = None,
) -> float:
    """
    Returns option premium per contract for entry.
    Long pays ask + slippage.
    Short receives bid - slippage.
    """
    cfg = config or FillConfig()
    slip = compute_leg_slippage(bid=bid, ask=ask, config=cfg.slippage)

    if side == "long":
        return max(ask + slip, 0.0)
    if side == "short":
        return max(bid - slip, 0.0)

    raise ValueError(f"Unknown side: {side}")


def exit_fill_price(
    *,
    side: str,
    bid: float,
    ask: float,
    config: FillConfig | None = None,
) -> float:
    """
    Returns option premium per contract for exit.
    Long exits by selling at bid - slippage.
    Short exits by buying back at ask + slippage.
    """
    cfg = config or FillConfig()
    slip = compute_leg_slippage(bid=bid, ask=ask, config=cfg.slippage)

    if side == "long":
        return max(bid - slip, 0.0)
    if side == "short":
        return max(ask + slip, 0.0)

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
