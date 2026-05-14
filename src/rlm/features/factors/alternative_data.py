from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.factors.base import FactorCalculator
from rlm.types.factors import FactorCategory, FactorSpec, TransformKind


class AlternativeDataFactors(FactorCalculator):
    def specs(self) -> list[FactorSpec]:
        return [
            FactorSpec("macro_sentiment", FactorCategory.DIRECTION, TransformKind.SIGNED, scale_value=0.5),
            FactorSpec("fed_speech_sentiment", FactorCategory.DIRECTION, TransformKind.SIGNED, scale_value=0.5),
            FactorSpec("vix_futures_curve", FactorCategory.VOLATILITY, TransformKind.SIGNED, scale_value=0.1),
            FactorSpec("credit_spread", FactorCategory.LIQUIDITY, TransformKind.RATIO, neutral_value=0.0),
        ]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in ("macro_sentiment", "fed_speech_sentiment", "vix_futures_curve", "credit_spread"):
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        return out
