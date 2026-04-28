"""ForecastService — application layer for the end-to-end forecast pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.config import build_pipeline_config
from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.utils.logging import get_logger
from rlm.utils.timing import timed_stage

log = get_logger(__name__)

# Columns surfaced to the user by default
_OUTPUT_COLUMNS = [
    "close",
    "S_D",
    "S_V",
    "S_L",
    "S_G",
    "b_m",
    "b_sigma",
    "mu",
    "sigma",
    "mean_price",
    "lower_1s",
    "upper_1s",
    "lower_2s",
    "upper_2s",
    "realized_vol",
    "forecast_source",
    "hmm_state",
    "hmm_state_label",
    "hmm_confidence",
    "hmm_regime_transition_entropy",
    "hmm_expected_persistence",
    "hmm_most_likely_next_state",
    "hmm_most_likely_next_prob",
    "hmm_most_likely_next_label",
    "markov_state",
    "markov_state_label",
    "kronos_confidence",
    "kronos_regime_agreement",
    "kronos_predicted_regime",
    "kronos_transition_flag",
    "kronos_forecast_return",
    "kronos_forecast_vol",
]


@dataclass
class ForecastRequest:
    """Input bundle for a single forecast run."""

    symbol: str
    bars_df: pd.DataFrame
    option_chain_df: pd.DataFrame | None = None
    config: FullRLMConfig | None = None
    # Output control
    out_path: Path | None = None
    write_output: bool = True
    tail_rows: int = 10


@dataclass
class ForecastArtifacts:
    """Paths and metadata for files written by ForecastService."""

    forecast_csv: Path | None = None
    rows_written: int = 0
    duration_s: float = 0.0


class ForecastService:
    """Run the full factor → regime → ROEE forecast pipeline.

    This is the single authoritative path between the CLI / API surface and
    ``FullRLMPipeline``.  Orchestration, logging, column selection, and
    artifact writing all live here so CLI modules stay thin.
    """

    def run(self, req: ForecastRequest) -> PipelineResult:
        cfg = req.config or build_pipeline_config(symbol=req.symbol, overrides={})
        t0 = time.monotonic()

        log.info(
            "forecast start  symbol=%s regime=%s kronos=%s backtest=%s bars=%d",
            req.symbol,
            cfg.regime_model,
            cfg.use_kronos,
            cfg.run_backtest,
            len(req.bars_df),
        )

        pipeline = FullRLMPipeline(cfg)
        with timed_stage(log, "forecast_pipeline", symbol=req.symbol):
            result = pipeline.run(req.bars_df, req.option_chain_df)

        log.info(
            "forecast done  symbol=%s factors=%d forecast=%d policy=%d duration=%.1fs",
            req.symbol,
            len(result.factors_df),
            len(result.forecast_df),
            len(result.policy_df),
            time.monotonic() - t0,
        )
        return result

    def write_outputs(self, req: ForecastRequest, result: PipelineResult) -> ForecastArtifacts:
        """Write forecast CSV and return artifact metadata."""
        if not req.write_output or req.out_path is None:
            return ForecastArtifacts()

        t0 = time.monotonic()
        req.out_path.parent.mkdir(parents=True, exist_ok=True)
        result.forecast_df.to_csv(req.out_path)

        arts = ForecastArtifacts(
            forecast_csv=req.out_path,
            rows_written=len(result.forecast_df),
            duration_s=time.monotonic() - t0,
        )
        log.info("forecast output  path=%s rows=%d", arts.forecast_csv, arts.rows_written)
        return arts

    def summarize(self, result: PipelineResult) -> dict:
        """Return a human-readable summary dict from the pipeline result."""
        if result.forecast_df.empty:
            return {}

        available = [c for c in _OUTPUT_COLUMNS if c in result.forecast_df.columns]
        tail = result.forecast_df[available].tail(1)

        last_policy = result.policy_df.iloc[-1] if not result.policy_df.empty else None
        summary: dict = {
            "rows": len(result.forecast_df),
            "action": last_policy.get("roee_action") if last_policy is not None else None,
            "strategy": last_policy.get("roee_strategy") if last_policy is not None else None,
            "size_fraction": (
                last_policy.get("roee_size_fraction") if last_policy is not None else None
            ),
        }
        if result.backtest_metrics:
            summary["backtest"] = result.backtest_metrics
        return summary

    @staticmethod
    def output_columns(result: PipelineResult) -> list[str]:
        """Return the subset of standard output columns present in the result."""
        return [c for c in _OUTPUT_COLUMNS if c in result.forecast_df.columns]
