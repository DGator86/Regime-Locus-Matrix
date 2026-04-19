from __future__ import annotations

from rlm.training.execution_model import ExecutionAssumptions, normalized_slippage_penalty


def test_slippage_penalty_increases_with_vol_and_lower_liquidity() -> None:
    low = normalized_slippage_penalty(
        realized_vol=0.01,
        liquidity_score=8.0,
        width_fraction=0.01,
        assumptions=ExecutionAssumptions(),
    )
    high = normalized_slippage_penalty(
        realized_vol=0.05,
        liquidity_score=2.0,
        width_fraction=0.01,
        assumptions=ExecutionAssumptions(),
    )
    assert high > low
