"""Backward-compatibility re-export. Canonical location: rlm.features.scoring."""

from rlm.features.scoring.state_matrix import (
    classify_state_matrix,
    make_regime_key,
    regime_is_tradeable,
)
from rlm.features.scoring.thresholds import (
    classify_dealer_flow,
    classify_direction,
    classify_liquidity,
    classify_volatility,
)

__all__ = [
    "classify_state_matrix",
    "make_regime_key",
    "regime_is_tradeable",
    "classify_dealer_flow",
    "classify_direction",
    "classify_liquidity",
    "classify_volatility",
]
