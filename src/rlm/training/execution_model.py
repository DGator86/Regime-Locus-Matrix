from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionAssumptions:
    base_slippage_bps: float = 8.0
    vol_slippage_mult: float = 2.5
    illiquidity_slippage_mult: float = 1.5
    max_spread_penalty: float = 0.25
    exit_cost_mult: float = 1.15


def normalized_slippage_penalty(
    *,
    realized_vol: float,
    liquidity_score: float,
    width_fraction: float,
    assumptions: ExecutionAssumptions,
) -> float:
    """Estimate normalized friction in strategy-risk units for training targets."""
    vol_term = assumptions.base_slippage_bps / 10000.0
    vol_term *= 1.0 + assumptions.vol_slippage_mult * max(realized_vol, 0.0)

    illiq_term = max(0.0, 5.0 - liquidity_score) / 5.0
    illiq_term *= assumptions.illiquidity_slippage_mult * vol_term

    spread_term = min(max(width_fraction, 0.0) * 0.5, assumptions.max_spread_penalty)
    return float(vol_term + illiq_term + spread_term)
