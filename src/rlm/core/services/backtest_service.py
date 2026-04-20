"""BacktestService — application layer for backtesting and walk-forward runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestRequest:
    """Input bundle for a single backtest run."""

    symbol: str
    bars_df: pd.DataFrame
    option_chain_df: pd.DataFrame | None = None
    config: FullRLMConfig | None = None
    walkforward: bool = False
    out_dir: Path | None = None


class BacktestService:
    """Orchestrate a full pipeline run followed by backtesting.

    Wraps ``FullRLMPipeline`` with ``run_backtest=True`` and optionally
    chains a walk-forward validation pass.
    """

    def run(self, req: BacktestRequest) -> PipelineResult:
        cfg = req.config or FullRLMConfig(symbol=req.symbol)

        if not cfg.run_backtest:
            from dataclasses import replace
            cfg = replace(cfg, run_backtest=True)

        log.info(
            "BacktestService.run symbol=%s regime=%s walkforward=%s",
            req.symbol,
            cfg.regime_model,
            req.walkforward,
        )

        result = FullRLMPipeline(cfg).run(req.bars_df, req.option_chain_df)

        if req.walkforward and result.policy_df is not None:
            result = self._run_walkforward(req, result)

        if result.backtest_metrics:
            log.info("Backtest metrics: %s", result.backtest_metrics)

        return result

    def _run_walkforward(self, req: BacktestRequest, base_result: PipelineResult) -> PipelineResult:
        """Run walk-forward validation using the existing policy_df."""
        try:
            from rlm.backtest.walkforward import WalkForwardEngine, WalkForwardConfig

            cfg = req.config or FullRLMConfig(symbol=req.symbol)
            wf_cfg = WalkForwardConfig()
            engine = WalkForwardEngine(pipeline_config=cfg, wf_config=wf_cfg)
            wf_result = engine.run(req.bars_df, req.option_chain_df)
            log.info("Walk-forward complete: %d windows", len(wf_result.window_results))
        except Exception as exc:
            log.warning("Walk-forward failed: %s", exc)

        return base_result
