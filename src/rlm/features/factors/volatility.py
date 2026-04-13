from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class VolatilityFactors(FactorCalculator):
    """
    Expected columns where available:
      close, high, low
      vix, vvix
    """

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="atr_over_price",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.01,
                k=0.8,
            ),
            FactorSpec(
                name="bollinger_width",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.03,
                k=0.9,
            ),
            FactorSpec(
                name="realized_volatility",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.01,
                k=0.8,
            ),
            FactorSpec(
                name="vix_ratio",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="vvix_ratio",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.8,
            ),
            FactorSpec(
                name="abnormal_options_volume",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="options_volume_to_oi",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.15,
                k=0.9,
            ),
            FactorSpec(
                name="underlying_vix_corr",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.25,
                k=1.0,
            ),
            FactorSpec(
                name="underlying_vix_corr_breakdown",
                category=FactorCategory.VOLATILITY,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.15,
                k=1.0,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean()

        out["atr_over_price"] = atr / close

        ma20 = close.rolling(20, min_periods=5).mean()
        std20 = close.rolling(20, min_periods=5).std()
        upper = ma20 + 2.0 * std20
        lower = ma20 - 2.0 * std20
        out["bollinger_width"] = (upper - lower) / ma20

        ret = close.pct_change()
        out["realized_volatility"] = ret.rolling(20, min_periods=5).std()

        if "options_volume" in df.columns:
            options_volume_avg = df["options_volume"].rolling(20, min_periods=5).mean().replace(0, np.nan)
            out["abnormal_options_volume"] = df["options_volume"] / options_volume_avg
        else:
            out["abnormal_options_volume"] = pd.NA

        if "options_volume_to_oi" in df.columns:
            out["options_volume_to_oi"] = df["options_volume_to_oi"]
        else:
            out["options_volume_to_oi"] = pd.NA

        if "vix" in df.columns:
            vix_med = df["vix"].rolling(252, min_periods=20).median()
            out["vix_ratio"] = df["vix"] / vix_med
            vix_ret = df["vix"].pct_change()
            corr20 = ret.rolling(20, min_periods=10).corr(vix_ret)
            out["underlying_vix_corr"] = corr20
            corr_baseline = corr20.rolling(126, min_periods=20).median()
            out["underlying_vix_corr_breakdown"] = corr20 - corr_baseline
        else:
            out["vix_ratio"] = pd.NA
            out["underlying_vix_corr"] = pd.NA
            out["underlying_vix_corr_breakdown"] = pd.NA

        if "vvix" in df.columns:
            vvix_med = df["vvix"].rolling(252, min_periods=20).median()
            out["vvix_ratio"] = df["vvix"] / vvix_med
        else:
            out["vvix_ratio"] = pd.NA

        return out
