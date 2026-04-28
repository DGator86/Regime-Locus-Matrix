from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rlm.roee.strategy_value_model import STRATEGY_NAMES


def simulate_strategy_target_row_v1(
    row: pd.Series,
    forward_df: pd.DataFrame,
    strike_increment: float,
) -> dict[str, float]:
    """Phase-1 deterministic target simulator for coordinate strategy training."""
    if forward_df.empty:
        return {name: 0.0 for name in STRATEGY_NAMES}

    start_close = _safe_float(row, "close", default=np.nan)
    end_close = _safe_float(forward_df.iloc[-1], "close", default=np.nan)
    if not np.isfinite(start_close) or start_close <= 0.0 or not np.isfinite(end_close):
        return {name: 0.0 for name in STRATEGY_NAMES}

    realized_move = (end_close - start_close) / start_close
    realized_vol = float(forward_df["close"].pct_change().std(ddof=0)) if len(forward_df) > 1 else 0.0
    if not np.isfinite(realized_vol):
        realized_vol = 0.0

    sigma = _safe_float(row, "sigma", default=max(realized_vol, 1e-6))
    transition = abs(_safe_float(row, "M_R_trans", default=0.0))
    trend = _safe_float(row, "M_trend_strength", default=0.0)

    width = max(strike_increment / start_close, 1e-4)
    slippage_penalty = 0.02 + 0.2 * realized_vol + 0.1 * width
    instability_penalty = 0.01 * transition

    direction_edge = realized_move / max(sigma, 1e-4)
    range_edge = max(width - abs(realized_move), -width) / max(width, 1e-4)

    targets: dict[str, float] = {}
    targets["bull_call_spread"] = direction_edge - slippage_penalty - 0.5 * instability_penalty
    targets["bear_put_spread"] = -direction_edge - slippage_penalty - 0.5 * instability_penalty
    targets["iron_condor"] = range_edge - slippage_penalty - 0.3 * abs(direction_edge)
    targets["calendar_spread"] = (
        (realized_vol - sigma) / max(sigma, 1e-4) - slippage_penalty - 0.25 * instability_penalty
    )
    targets["debit_spread"] = abs(direction_edge) + 0.05 * trend - slippage_penalty - instability_penalty
    targets["no_trade"] = 0.0

    return {name: float(targets.get(name, 0.0)) for name in STRATEGY_NAMES}


def _safe_float(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default
