from __future__ import annotations

import pandas as pd

from rlm.factors.base import compute_composite_scores, standardize_factor_frame
from rlm.factors.config import filter_specs, load_feature_engineering_config
from rlm.factors.candle_patterns import CandlePatternFactors
from rlm.factors.dealer_flow import DealerFlowFactors
from rlm.factors.liquidity import LiquidityFactors
from rlm.factors.order_flow import OrderFlowFactors
from rlm.factors.volatility import VolatilityFactors
from rlm.types.factors import FactorSpec


class FactorPipeline:
    def __init__(self, *, feature_config: dict[str, object] | None = None) -> None:
        self.feature_config = (
            load_feature_engineering_config() if feature_config is None else feature_config
        )
        self.calculators = [
            OrderFlowFactors(),
            CandlePatternFactors(),
            VolatilityFactors(),
            LiquidityFactors(),
            DealerFlowFactors(),
        ]

    def specs(self) -> list[FactorSpec]:
        specs: list[FactorSpec] = []
        for calc in self.calculators:
            specs.extend(calc.specs())
        return filter_specs(specs, self.feature_config)

    def compute_raw_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for calc in self.calculators:
            frames.append(calc.compute(df))
        raw_factors = pd.concat(frames, axis=1)
        enabled_names = [spec.name for spec in self.specs()]
        return raw_factors.reindex(columns=enabled_names)

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
