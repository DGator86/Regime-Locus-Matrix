from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rlm.roee.strategy_value_model import STRATEGY_NAMES


def simulate_strategy_target_row_v2(
    row: pd.Series,
    forward_df: pd.DataFrame,
    strike_increment: float,
    horizon: int,
    use_path_exits: bool = True,
) -> dict[str, float]:
    """Path- and structure-aware strategy target approximation for training."""
    if forward_df.empty or horizon <= 0:
        return {name: 0.0 for name in STRATEGY_NAMES}

    start = _safe_float(row, "close", np.nan)
    if not np.isfinite(start) or start <= 0:
        return {name: 0.0 for name in STRATEGY_NAMES}

    closes = forward_df["close"].astype(float).to_numpy(copy=False)
    if closes.size == 0 or not np.isfinite(closes).all():
        return {name: 0.0 for name in STRATEGY_NAMES}

    sigma0 = max(_safe_float(row, "sigma", 0.01), 1e-4)
    p_tau = float(closes[-1])
    realized_move = (p_tau - start) / start
    rv = float(np.std(np.diff(np.log(np.maximum(closes, 1e-8))), ddof=0)) if closes.size > 1 else 0.0

    width_abs = max(strike_increment, 0.25)
    width = max(width_abs / start, 1e-4)
    transition = abs(_safe_float(row, "M_R_trans", 0.0))
    trend = _safe_float(row, "M_trend_strength", 0.0)

    slippage = 0.01 + 0.05 * rv + 0.04 * width
    illiquidity_penalty = 0.01 * max(0.0, 5.0 - _safe_float(row, "M_L", 5.0))
    transition_penalty = 0.02 * transition

    k_long_call = start
    k_short_call = start + width_abs
    call_debit = max(0.2 * width_abs, 1e-3)
    bull_intrinsic = min(max(p_tau - k_long_call, 0.0), max(k_short_call - k_long_call, 1e-6))
    bull = (bull_intrinsic - call_debit) / max(call_debit, 1e-6) - slippage - 0.5 * illiquidity_penalty

    k_long_put = start
    k_short_put = start - width_abs
    put_debit = max(0.2 * width_abs, 1e-3)
    bear_intrinsic = min(max(k_long_put - p_tau, 0.0), max(k_long_put - k_short_put, 1e-6))
    bear = (bear_intrinsic - put_debit) / max(put_debit, 1e-6) - slippage - 0.5 * illiquidity_penalty

    corridor_low = start - width_abs
    corridor_high = start + width_abs
    if use_path_exits:
        breach = float(np.mean((closes < corridor_low) | (closes > corridor_high)))
    else:
        breach = float(p_tau < corridor_low or p_tau > corridor_high)
    credit_efficiency = max(0.05, 0.3 - 0.7 * rv)
    condor = credit_efficiency - 1.2 * breach - slippage - 0.3 * abs(realized_move / sigma0)

    displacement = abs(p_tau - start) / start
    calendar = (
        1.1 * ((rv - sigma0) / max(sigma0, 1e-4))
        - 0.8 * displacement
        - slippage
        - 0.25 * illiquidity_penalty
    )

    direction_edge = realized_move / sigma0
    trend_sign = np.sign(trend) if trend != 0 else np.sign(_safe_float(row, "M_D", 5.0) - 5.0)
    direction_match = np.sign(realized_move) * trend_sign
    debit = abs(direction_edge) + 0.1 * abs(trend)
    if direction_match < 0:
        debit = -debit
    debit = debit - slippage - illiquidity_penalty - transition_penalty

    scores = {
        "bull_call_spread": float(bull),
        "bear_put_spread": float(bear),
        "iron_condor": float(condor),
        "calendar_spread": float(calendar),
        "debit_spread": float(debit),
        "no_trade": 0.0,
    }
    return {name: float(scores.get(name, 0.0)) for name in STRATEGY_NAMES}


def _safe_float(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default
