from __future__ import annotations

from typing import Iterable

import pandas as pd

from rlm.features.scoring.coordinate_regime_bootstrap import (
    bootstrap_regime_label_from_coordinates,
)
from rlm.roee.strategy_value_model import STRATEGY_NAMES
from rlm.training.strategy_targets import simulate_strategy_target_row

REQUIRED_COORD_COLUMNS: tuple[str, ...] = (
    "M_D",
    "M_V",
    "M_L",
    "M_G",
    "M_trend_strength",
    "M_dealer_control",
    "M_alignment",
    "M_delta_neutral",
    "M_R_trans",
)

_OPTIONAL_METADATA_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "close",
    "sigma",
    "direction_regime",
    "volatility_regime",
    "liquidity_regime",
    "dealer_flow_regime",
    "regime_key",
)


def build_regime_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_COORD_COLUMNS)
    cols = [c for c in _OPTIONAL_METADATA_COLUMNS if c in df.columns] + list(REQUIRED_COORD_COLUMNS)
    out = df.loc[:, cols].copy()
    out["regime_label"] = out.apply(bootstrap_regime_label_from_coordinates, axis=1)
    return out.dropna(subset=list(REQUIRED_COORD_COLUMNS)).reset_index(drop=True)


def build_strategy_value_training_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_COORD_COLUMNS)
    if "close" not in df.columns:
        raise ValueError("build_strategy_value_training_frame requires a 'close' column")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    cols = [c for c in _OPTIONAL_METADATA_COLUMNS if c in df.columns] + list(REQUIRED_COORD_COLUMNS)
    base = df.loc[:, cols].copy().reset_index(drop=True)

    target_rows: list[dict[str, float]] = []
    last_train_idx = max(len(base) - horizon, 0)
    for idx, row in base.iloc[:last_train_idx].iterrows():
        forward_df = base.iloc[idx + 1 : idx + horizon + 1]
        target_rows.append(simulate_strategy_target_row(row, forward_df, strike_increment=5.0))

    out = base.iloc[:last_train_idx].copy()
    targets_df = pd.DataFrame(target_rows, index=out.index)
    for col in STRATEGY_NAMES:
        out[col] = targets_df.get(col, 0.0)
    out["no_trade"] = 0.0
    return out.reset_index(drop=True)


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")
