"""ForecastService — application layer for the end-to-end forecast pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ForecastRequest:
    """Input bundle for a single forecast run."""

    symbol: str
    bars_df: pd.DataFrame
    option_chain_df: pd.DataFrame | None = None
    config: FullRLMConfig | None = None


class ForecastService:
    """Run the full factor → regime → ROEE forecast pipeline.

    This is the single authoritative path between the CLI / API surface and
    ``FullRLMPipeline``.  All input preparation, logging, and error handling
    live here so callers stay thin.
    """

    def run(self, req: ForecastRequest) -> PipelineResult:
        cfg = req.config or FullRLMConfig(symbol=req.symbol)

        log.info(
            "ForecastService.run symbol=%s regime=%s kronos=%s backtest=%s",
            req.symbol,
            cfg.regime_model,
            cfg.use_kronos,
            cfg.run_backtest,
        )

        pipeline = FullRLMPipeline(cfg)
        result = pipeline.run(req.bars_df, req.option_chain_df)

        log.info(
            "ForecastService done: factors=%d forecast=%d policy=%d",
            len(result.factors_df),
            len(result.forecast_df),
            len(result.policy_df),
        )
        return result
