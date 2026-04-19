from __future__ import annotations

from typing import Iterable

import pandas as pd

from rlm.features.scoring.coordinate_regime_bootstrap import (
    bootstrap_regime_label_from_coordinates,
)
from rlm.roee.strategy_value_model import STRATEGY_NAMES
from rlm.training.regime_targets import derive_outcome_regime_label
from rlm.training.strategy_targets_v1 import simulate_strategy_target_row_v1
from rlm.training.strategy_targets_v2 import simulate_strategy_target_row_v2

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


def build_regime_training_frame(
    df: pd.DataFrame,
    *,
    label_mode: str = "bootstrap",
    horizon: int = 20,
) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_COORD_COLUMNS)
    cols = [c for c in _OPTIONAL_METADATA_COLUMNS if c in df.columns] + list(REQUIRED_COORD_COLUMNS)
    out = df.loc[:, cols].copy().reset_index(drop=True)

    if label_mode == "bootstrap":
        out["regime_label"] = out.apply(bootstrap_regime_label_from_coordinates, axis=1)
        return out.dropna(subset=list(REQUIRED_COORD_COLUMNS)).reset_index(drop=True)

    if label_mode != "outcome":
        raise ValueError("label_mode must be one of: bootstrap, outcome")
    if "close" not in out.columns:
        raise ValueError("build_regime_training_frame(..., label_mode='outcome') requires 'close'")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    target_rows = _build_target_rows(out, horizon=horizon, target_mode="v2")
    usable = out.iloc[: len(target_rows)].copy()
    targets_df = pd.DataFrame(target_rows, index=usable.index)
    usable["regime_label"] = targets_df.apply(derive_outcome_regime_label, axis=1)
    return usable.dropna(subset=list(REQUIRED_COORD_COLUMNS)).reset_index(drop=True)


def build_strategy_value_training_frame(
    df: pd.DataFrame,
    horizon: int,
    *,
    target_mode: str = "v2",
) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_COORD_COLUMNS)
    if "close" not in df.columns:
        raise ValueError("build_strategy_value_training_frame requires a 'close' column")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if target_mode not in {"v1", "v2"}:
        raise ValueError("target_mode must be one of: v1, v2")

    cols = [c for c in _OPTIONAL_METADATA_COLUMNS if c in df.columns] + list(REQUIRED_COORD_COLUMNS)
    base = df.loc[:, cols].copy().reset_index(drop=True)

    target_rows = _build_target_rows(base, horizon=horizon, target_mode=target_mode)
    out = base.iloc[: len(target_rows)].copy()
    targets_df = pd.DataFrame(target_rows, index=out.index)
    for col in STRATEGY_NAMES:
        out[col] = targets_df.get(col, 0.0)
    out["no_trade"] = 0.0
    return out.reset_index(drop=True)


def _build_target_rows(base: pd.DataFrame, *, horizon: int, target_mode: str) -> list[dict[str, float]]:
    target_rows: list[dict[str, float]] = []
    last_train_idx = max(len(base) - horizon, 0)
    for idx, row in base.iloc[:last_train_idx].iterrows():
        forward_df = base.iloc[idx + 1 : idx + horizon + 1]
        if target_mode == "v1":
            target = simulate_strategy_target_row_v1(row, forward_df, strike_increment=5.0)
        else:
            target = simulate_strategy_target_row_v2(
                row,
                forward_df,
                strike_increment=5.0,
                horizon=horizon,
                use_path_exits=True,
            )
        target_rows.append(target)
    return target_rows


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")
