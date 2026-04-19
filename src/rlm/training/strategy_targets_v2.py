from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from rlm.roee.strategy_value_model import STRATEGY_NAMES
from rlm.training.execution_model import ExecutionAssumptions, normalized_slippage_penalty
from rlm.training.strategy_structures import (
    build_bear_put_structure,
    build_bull_call_structure,
    build_calendar_structure,
    build_debit_spread_structure,
    build_iron_condor_structure,
)
from rlm.training.strategy_valuation import (
    value_bear_put_at_expiry,
    value_bull_call_at_expiry,
    value_calendar_path,
    value_debit_spread_path,
    value_iron_condor_path,
)


def simulate_strategy_target_row_v2(
    row: pd.Series,
    forward_df: pd.DataFrame,
    strike_increment: float,
    horizon: int,
    use_path_exits: bool = True,
) -> dict[str, float]:
    if forward_df.empty or horizon <= 0:
        return {name: 0.0 for name in STRATEGY_NAMES}

    start = _safe_float(row, "close", np.nan)
    path = forward_df["close"].astype(float).to_numpy(copy=False)
    if len(path) == 0 or not np.isfinite(path).all() or start <= 0:
        return {name: 0.0 for name in STRATEGY_NAMES}

    entry_sigma = max(_safe_float(row, "sigma", 0.01), 1e-4)
    liquidity_score = _safe_float(row, "M_L", 5.0)
    trend_strength = abs(_safe_float(row, "M_trend_strength", 0.0))
    realized_vol = (
        float(np.std(np.diff(np.log(np.maximum(path, 1e-8))), ddof=0)) if len(path) > 1 else 0.0
    )
    final_price = float(path[-1])

    width_abs = max(strike_increment, 0.25)
    width_fraction = width_abs / start
    assumptions = ExecutionAssumptions()

    bull = build_bull_call_structure(start, width_abs)
    bear = build_bear_put_structure(start, width_abs)
    condor = build_iron_condor_structure(start, width_abs)
    calendar = build_calendar_structure(start, width_abs)
    debit = build_debit_spread_structure(
        start,
        width_abs,
        bias=np.sign(_safe_float(row, "M_D", 5.0) - 5.0),
    )

    slippage = normalized_slippage_penalty(
        realized_vol=realized_vol,
        liquidity_score=liquidity_score,
        width_fraction=width_fraction,
        assumptions=assumptions,
    )

    bull_value = value_bull_call_at_expiry(bull, final_price)
    bear_value = value_bear_put_at_expiry(bear, final_price)
    condor_value = value_iron_condor_path(condor, path, use_path_exits=use_path_exits)
    calendar_value = value_calendar_path(
        calendar,
        path,
        entry_sigma=entry_sigma,
        realized_vol=realized_vol,
        horizon_fraction=min(horizon / 30.0, 1.0),
    )
    debit_value = value_debit_spread_path(debit, path, trend_strength=trend_strength)

    transition_penalty = 0.01 * abs(_safe_float(row, "M_R_trans", 0.0))
    scores = {
        "bull_call_spread": _normalize_target(
            bull_value, bull.max_risk, bull.entry_price, slippage, 0.0
        ),
        "bear_put_spread": _normalize_target(
            bear_value, bear.max_risk, bear.entry_price, slippage, 0.0
        ),
        "iron_condor": _normalize_target(
            condor_value,
            condor.max_risk,
            0.0,
            slippage * assumptions.exit_cost_mult,
            transition_penalty,
        ),
        "calendar_spread": _normalize_target(
            calendar_value,
            calendar.max_risk,
            calendar.entry_price,
            slippage,
            0.0,
        ),
        "debit_spread": _normalize_target(
            debit_value,
            debit.max_risk,
            debit.entry_price,
            slippage,
            transition_penalty,
        ),
        "no_trade": 0.0,
    }
    return {name: float(scores.get(name, 0.0)) for name in STRATEGY_NAMES}


def _normalize_target(
    gross_value: float,
    max_risk: float,
    entry_price: float,
    slippage_penalty: float,
    extra_penalty: float,
) -> float:
    pnl = gross_value - entry_price
    return float((pnl / max(max_risk, 1e-6)) - slippage_penalty - extra_penalty)


def _safe_float(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default
