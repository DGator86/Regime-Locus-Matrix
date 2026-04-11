from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - polars is optional at runtime
    pl = None

from rlm.factors.base import compute_composite_scores, standardize_factor_frame
from rlm.factors.config import filter_specs, load_feature_engineering_config
from rlm.factors.candle_patterns import CandlePatternFactors
from rlm.factors.dealer_flow import DealerFlowFactors
from rlm.factors.kronos_factors import KronosFactorCalculator
from rlm.factors.liquidity import LiquidityFactors
from rlm.factors.multi_timeframe_engine import MultiTimeframeEngine
from rlm.factors.multi_timeframe_liquidity import MultiTimeframeLiquidityFactors
from rlm.factors.liquidity_pools import AdvancedLiquidityPoolFactors
from rlm.factors.order_flow import OrderFlowFactors
from rlm.factors.support_resistance import SupportResistanceFactors
from rlm.factors.volatility import VolatilityFactors
from rlm.types.factors import FactorSpec
from rlm.utils.parallel import parallel_map


def _compute_factor_frame(task: tuple[Any, pd.DataFrame]) -> pd.DataFrame:
    calc, df = task
    return calc.compute(df)


class FactorPipeline:
    def __init__(
        self,
        *,
        feature_config: dict[str, object] | None = None,
        max_workers: int | None = None,
        parallel_backend: str = "thread",
    ) -> None:
        self.feature_config = (
            load_feature_engineering_config() if feature_config is None else feature_config
        )
        base_calculators = [
            OrderFlowFactors(),
            VolatilityFactors(),
            LiquidityFactors(),
            DealerFlowFactors(),
        ]
        self.calculators = [MultiTimeframeEngine(calc) for calc in base_calculators] + [
            CandlePatternFactors(),
            SupportResistanceFactors(),
            MultiTimeframeLiquidityFactors(),
            AdvancedLiquidityPoolFactors(),
            KronosFactorCalculator(),
        ]
        self.max_workers = (
            max_workers if max_workers is not None else int(os.getenv("RLM_FACTOR_WORKERS", "1"))
        )
        self.parallel_backend = str(parallel_backend)

    def specs(self) -> list[FactorSpec]:
        specs: list[FactorSpec] = []
        for calc in self.calculators:
            specs.extend(calc.specs())
        return filter_specs(specs, self.feature_config)

    def _concat_factor_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        if pl is None:
            return pd.concat(frames, axis=1)

        idx_name = "__rlm_idx__"
        normalized = []
        for frame in frames:
            part = frame.copy()
            part[idx_name] = part.index
            normalized.append(pl.from_pandas(part.reset_index(drop=True)))

        merged = normalized[0]
        for frame in normalized[1:]:
            merged = merged.join(frame, on=idx_name, how="inner")

        out = merged.to_pandas().set_index(idx_name)
        out.index.name = frames[0].index.name
        return out

    def compute_raw_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        tasks = [(calc, df) for calc in self.calculators]
        frames = parallel_map(
            _compute_factor_frame,
            tasks,
            max_workers=self.max_workers,
            backend=self.parallel_backend,
        )
        raw_factors = self._concat_factor_frames(frames)
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
