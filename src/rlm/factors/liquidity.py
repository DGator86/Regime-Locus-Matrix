from __future__ import annotations

import pandas as pd

from rlm.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class LiquidityFactors(FactorCalculator):
    """
    Expected columns where available:
      close, volume
      bid_ask_spread
      order_book_depth
    """

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="spread_over_price",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.0005,
                k=1.2,
                invert=True,
            ),
            FactorSpec(
                name="volume_vs_average",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="order_book_depth_ratio",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        close = df["close"]
        volume = df["volume"]

        if "bid_ask_spread" in df.columns:
            out["spread_over_price"] = df["bid_ask_spread"] / close
        else:
            out["spread_over_price"] = pd.NA

        vol_avg = volume.rolling(20, min_periods=5).mean()
        out["volume_vs_average"] = volume / vol_avg

        if "order_book_depth" in df.columns:
            depth_avg = df["order_book_depth"].rolling(20, min_periods=5).mean()
            out["order_book_depth_ratio"] = df["order_book_depth"] / depth_avg
        else:
            out["order_book_depth_ratio"] = pd.NA

        return out
