from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class SupportResistanceFactors(FactorCalculator):
    """
    Support/resistance zone factors derived from rolling swing highs/lows.

    Expected columns:
      close, high, low, volume
    """

    def __init__(
        self,
        *,
        swing_window: int = 20,
        atr_window: int = 14,
        touch_window: int = 60,
        touch_atr_threshold: float = 0.25,
        volume_confirm_ratio: float = 1.2,
        breakout_atr_threshold: float = 0.15,
    ) -> None:
        self.swing_window = swing_window
        self.atr_window = atr_window
        self.touch_window = touch_window
        self.touch_atr_threshold = touch_atr_threshold
        self.volume_confirm_ratio = volume_confirm_ratio
        self.breakout_atr_threshold = breakout_atr_threshold

        self._specs = [
            FactorSpec(
                name="dist_to_nearest_support",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
                invert=True,
            ),
            FactorSpec(
                name="dist_to_nearest_resistance",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.RATIO,
                neutral_value=1.0,
                k=0.9,
            ),
            FactorSpec(
                name="zone_strength",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.RATIO,
                neutral_value=0.5,
                k=1.0,
            ),
            FactorSpec(
                name="breakout_confirmed",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=0.5,
                k=1.1,
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

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = (
            tr.rolling(self.atr_window, min_periods=max(5, self.atr_window // 2))
            .mean()
            .replace(0, np.nan)
        )

        resistance = high.rolling(
            self.swing_window, min_periods=max(5, self.swing_window // 2)
        ).max()
        support = low.rolling(self.swing_window, min_periods=max(5, self.swing_window // 2)).min()

        out["dist_to_nearest_support"] = ((close - support).clip(lower=0.0)) / atr
        out["dist_to_nearest_resistance"] = ((resistance - close).clip(lower=0.0)) / atr

        touch_band = atr * self.touch_atr_threshold
        volume_ma = volume.rolling(20, min_periods=5).mean()
        volume_confirmed = volume > (volume_ma * self.volume_confirm_ratio)

        support_touch = ((close - support).abs() <= touch_band) & volume_confirmed
        resistance_touch = ((close - resistance).abs() <= touch_band) & volume_confirmed

        support_touches = support_touch.rolling(self.touch_window, min_periods=5).sum()
        resistance_touches = resistance_touch.rolling(self.touch_window, min_periods=5).sum()

        total_touches = support_touches + resistance_touches
        rolling_touch_cap = total_touches.rolling(20, min_periods=5).max().replace(0, np.nan)
        out["zone_strength"] = (total_touches / rolling_touch_cap).clip(lower=0.0, upper=1.0)

        prior_resistance = resistance.shift(1)
        prior_support = support.shift(1)
        breakout_up = (
            close > (prior_resistance + atr * self.breakout_atr_threshold)
        ) & volume_confirmed
        breakout_down = (
            close < (prior_support - atr * self.breakout_atr_threshold)
        ) & volume_confirmed

        out["breakout_confirmed"] = np.where(
            breakout_up,
            1.0,
            np.where(breakout_down, -1.0, 0.0),
        )

        return out
