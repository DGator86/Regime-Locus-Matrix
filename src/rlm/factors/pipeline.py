from __future__ import annotations

import pandas as pd

from rlm.factors.base import compute_composite_scores, standardize_factor_frame
from rlm.factors.dealer_flow import DealerFlowFactors
from rlm.factors.liquidity import LiquidityFactors
from rlm.factors.order_flow import OrderFlowFactors
from rlm.factors.volatility import VolatilityFactors
from rlm.types.factors import FactorSpec


class FactorPipeline:
    def __init__(self) -> None:
        self.calculators = [
            OrderFlowFactors(),
            VolatilityFactors(),
            LiquidityFactors(),
            DealerFlowFactors(),
        ]

    def specs(self) -> list[FactorSpec]:
        specs: list[FactorSpec] = []
        for calc in self.calculators:
            specs.extend(calc.specs())
        return specs

    def compute_raw_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for calc in self.calculators:
            frames.append(calc.compute(df))
        return pd.concat(frames, axis=1)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_factors = self.compute_raw_factors(df)
        specs = self.specs()
        standardized = standardize_factor_frame(raw_factors, specs)
        scores = compute_composite_scores(standardized, specs)

        out = pd.concat(
            [
                df.copy(),
                raw_factors.add_prefix("raw_"),
                standardized.add_prefix("std_"),
                scores,
            ],
            axis=1,
        )
        return out
