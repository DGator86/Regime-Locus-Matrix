from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class OrderFlowFactors(FactorCalculator):
    """
    Computes the direction block.

    Expected columns where available:
      close, high, low, volume
      buy_volume, sell_volume
      vwap
      anchored_vwap
      advancers, decliners
      index_return
    """

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="price_vs_vwap",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.0025,
                k=1.0,
            ),
            FactorSpec(
                name="price_vs_anchored_vwap",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.0030,
                k=1.0,
            ),
            FactorSpec(
                name="cvd_slope",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.15,
                k=0.9,
            ),
            FactorSpec(
                name="volume_imbalance",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.10,
                k=1.1,
            ),
            FactorSpec(
                name="ma_spread_over_atr",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.75,
                k=0.9,
            ),
            FactorSpec(
                name="adx_direction_bias",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=5.0,
                k=0.8,
            ),
            FactorSpec(
                name="roc_n",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.01,
                k=1.0,
            ),
            FactorSpec(
                name="market_breadth_ratio",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="relative_strength_vs_index",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.005,
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
        volume = df["volume"]

        vwap = df["vwap"] if "vwap" in df.columns else ((high + low + close) / 3.0)
        anchored_vwap = df["anchored_vwap"] if "anchored_vwap" in df.columns else vwap

        out["price_vs_vwap"] = (close - vwap) / vwap
        out["price_vs_anchored_vwap"] = (close - anchored_vwap) / anchored_vwap

        if {"buy_volume", "sell_volume"}.issubset(df.columns):
            total = (df["buy_volume"] + df["sell_volume"]).replace(0, np.nan)
            out["volume_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / total
            cvd = (df["buy_volume"] - df["sell_volume"]).cumsum()
            denom = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
            out["cvd_slope"] = cvd.diff() / denom
        else:
            out["volume_imbalance"] = np.nan
            out["cvd_slope"] = np.nan

        ma20 = close.rolling(20, min_periods=5).mean()
        ma50 = close.rolling(50, min_periods=10).mean()

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean().replace(0, np.nan)

        out["ma_spread_over_atr"] = (ma20 - ma50) / atr

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.rolling(14, min_periods=5).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14, min_periods=5).mean() / atr)
        out["adx_direction_bias"] = plus_di - minus_di

        out["roc_n"] = close.pct_change(10)

        if {"advancers", "decliners"}.issubset(df.columns):
            out["market_breadth_ratio"] = df["advancers"] / df["decliners"].replace(0, np.nan)
        else:
            out["market_breadth_ratio"] = np.nan

        if "index_return" in df.columns:
            out["relative_strength_vs_index"] = close.pct_change(10) - df["index_return"]
        else:
            out["relative_strength_vs_index"] = np.nan

        return out
