from __future__ import annotations

import pandas as pd

from rlm.training.strategy_targets_v1 import simulate_strategy_target_row_v1
from rlm.training.strategy_targets_v2 import simulate_strategy_target_row_v2


def simulate_strategy_target_row(
    row: pd.Series,
    forward_df: pd.DataFrame,
    strike_increment: float,
    horizon: int,
    use_path_exits: bool = True,
) -> dict[str, float]:
    """Default strategy target simulator (v2)."""
    return simulate_strategy_target_row_v2(
        row,
        forward_df,
        strike_increment=strike_increment,
        horizon=horizon,
        use_path_exits=use_path_exits,
    )


__all__ = [
    "simulate_strategy_target_row",
    "simulate_strategy_target_row_v1",
    "simulate_strategy_target_row_v2",
]
