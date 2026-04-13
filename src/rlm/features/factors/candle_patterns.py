from __future__ import annotations

import importlib

import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class CandlePatternFactors(FactorCalculator):
    """Candlestick-pattern directional factors computed with TA-Lib."""

    def __init__(self) -> None:
        self._specs = [
            FactorSpec(
                name="candle_bullish_reversal",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="candle_bearish_reversal",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="candle_continuation",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
            ),
            FactorSpec(
                name="doji_or_spinning_top",
                category=FactorCategory.DIRECTION,
                transform_kind=TransformKind.SIGNED,
                scale_value=1.0,
                k=1.0,
                invert=True,
            ),
        ]

    def specs(self) -> list[FactorSpec]:
        return self._specs

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if importlib.util.find_spec("talib") is None:
            out["candle_bullish_reversal"] = pd.NA
            out["candle_bearish_reversal"] = pd.NA
            out["candle_continuation"] = pd.NA
            out["doji_or_spinning_top"] = pd.NA
            return out

        talib = importlib.import_module("talib")

        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        engulfing = talib.CDLENGULFING(open_, high, low, close).astype(float) / 100.0
        hammer = talib.CDLHAMMER(open_, high, low, close).astype(float) / 100.0
        shooting_star = talib.CDLSHOOTINGSTAR(open_, high, low, close).astype(float) / 100.0
        morning_star = talib.CDLMORNINGSTAR(open_, high, low, close).astype(float) / 100.0
        evening_star = talib.CDLEVENINGSTAR(open_, high, low, close).astype(float) / 100.0

        out["candle_bullish_reversal"] = (
            engulfing.clip(lower=0) + hammer.clip(lower=0) + morning_star.clip(lower=0)
        )
        out["candle_bearish_reversal"] = (
            (-engulfing.clip(upper=0)) + shooting_star.clip(lower=0) + evening_star.clip(lower=0)
        )

        marubozu = talib.CDLMARUBOZU(open_, high, low, close).astype(float) / 100.0
        rising_falling_three = talib.CDLRISEFALL3METHODS(open_, high, low, close).astype(float) / 100.0
        out["candle_continuation"] = marubozu + rising_falling_three

        doji = talib.CDLDOJI(open_, high, low, close).abs().astype(float) / 100.0
        spinning_top = talib.CDLSPINNINGTOP(open_, high, low, close).abs().astype(float) / 100.0
        out["doji_or_spinning_top"] = (doji + spinning_top).clip(upper=1.0)

        return out
