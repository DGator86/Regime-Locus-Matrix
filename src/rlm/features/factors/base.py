from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rlm.features.standardization.transforms import log_tanh_ratio, log_tanh_signed
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class FactorCalculator(ABC):
    @abstractmethod
    def specs(self) -> list[FactorSpec]:
        raise NotImplementedError

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def rolling_median_abs(series: pd.Series, window: int) -> pd.Series:
    return series.abs().rolling(window=window, min_periods=max(5, window // 3)).median()


def standardize_factor_series(
    series: pd.Series,
    spec: FactorSpec,
) -> pd.Series:
    if spec.transform_kind == TransformKind.RATIO:
        if spec.neutral_value is None:
            raise ValueError(f"{spec.name}: neutral_value required for ratio transform.")
        return series.apply(
            lambda x: log_tanh_ratio(
                x=x,
                x0=spec.neutral_value,
                k=spec.k,
                invert=spec.invert,
            )
        )

    if spec.transform_kind == TransformKind.SIGNED:
        if spec.scale_value is None:
            raise ValueError(f"{spec.name}: scale_value required for signed transform.")
        return series.apply(
            lambda x: log_tanh_signed(
                d=x,
                d0=spec.scale_value,
                k=spec.k,
                invert=spec.invert,
            )
        )

    raise ValueError(f"Unsupported transform kind: {spec.transform_kind}")


def standardize_factor_frame(
    raw_factors: pd.DataFrame,
    specs: list[FactorSpec],
) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_factors.index)
    for spec in specs:
        if spec.name not in raw_factors.columns:
            out[spec.name] = np.nan
            continue
        out[spec.name] = standardize_factor_series(raw_factors[spec.name], spec)
    return out


def compute_composite_scores(
    standardized_factors: pd.DataFrame,
    specs: list[FactorSpec],
) -> pd.DataFrame:
    category_to_names: dict[FactorCategory, list[str]] = {
        FactorCategory.DIRECTION: [],
        FactorCategory.VOLATILITY: [],
        FactorCategory.LIQUIDITY: [],
        FactorCategory.DEALER_FLOW: [],
    }

    for spec in specs:
        category_to_names[spec.category].append(spec.name)

    score_df = pd.DataFrame(index=standardized_factors.index)
    score_df["S_D"] = standardized_factors[category_to_names[FactorCategory.DIRECTION]].mean(
        axis=1, skipna=True
    )
    score_df["S_V"] = standardized_factors[category_to_names[FactorCategory.VOLATILITY]].mean(
        axis=1, skipna=True
    )
    score_df["S_L"] = standardized_factors[category_to_names[FactorCategory.LIQUIDITY]].mean(
        axis=1, skipna=True
    )
    score_df["S_G"] = (
        standardized_factors[category_to_names[FactorCategory.DEALER_FLOW]]
        .mean(axis=1, skipna=True)
        .fillna(0.0)
    )
    return score_df
