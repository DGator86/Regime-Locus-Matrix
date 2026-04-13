from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
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
                name="spread_pct_mid",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.08,
                k=1.0,
                invert=True,
            ),
            FactorSpec(
                name="spread_shock",
                category=FactorCategory.LIQUIDITY,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=1.0,
                invert=True,
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

        if "options_spread_pct_mid" in df.columns:
            out["spread_pct_mid"] = df["options_spread_pct_mid"]
        else:
            out["spread_pct_mid"] = np.nan

        if "bid_ask_spread" in df.columns:
            spread_base = df["bid_ask_spread"].rolling(20, min_periods=5).median().replace(0, np.nan)
            out["spread_shock"] = df["bid_ask_spread"] / spread_base
        else:
            out["spread_shock"] = np.nan

        if "order_book_depth" in df.columns:
            depth_avg = df["order_book_depth"].rolling(20, min_periods=5).mean()
            out["order_book_depth_ratio"] = df["order_book_depth"] / depth_avg
        else:
            out["order_book_depth_ratio"] = pd.NA

        return out
