"""ForecastService — application layer for the end-to-end forecast pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.core.run_manifest import RunManifest, build_config_summary, new_run_id, now_utc
from rlm.utils.logging import get_logger
from rlm.utils.timing import timed_stage

log = get_logger(__name__)

# Columns surfaced to the user by default
_OUTPUT_COLUMNS = [
    "close", "S_D", "S_V", "S_L", "S_G",
    "b_m", "b_sigma", "mu", "sigma",
    "mean_price", "lower_1s", "upper_1s", "lower_2s", "upper_2s",
    "realized_vol", "forecast_source",
    "hmm_state", "hmm_state_label",
    "markov_state", "markov_state_label",
    "kronos_confidence", "kronos_regime_agreement",
    "kronos_predicted_regime", "kronos_transition_flag",
    "kronos_forecast_return", "kronos_forecast_vol",
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
    # Run tracking
    run_id: str = field(default_factory=new_run_id)
    data_root: str | None = None
    backend: str = "auto"
    profile: str | None = None


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
        cfg = req.config or FullRLMConfig(symbol=req.symbol)

        with timed_stage(
            log, "forecast",
            run_id=req.run_id, symbol=req.symbol,
            regime=cfg.regime_model, kronos=cfg.use_kronos,
        ):
            pipeline = FullRLMPipeline(cfg)
            result = pipeline.run(req.bars_df, req.option_chain_df)

        log.info(
            "forecast rows  run_id=%s factors=%d forecast=%d policy=%d",
            req.run_id, len(result.factors_df), len(result.forecast_df), len(result.policy_df),
        )
        return result

    def write_outputs(
        self, req: ForecastRequest, result: PipelineResult
    ) -> ForecastArtifacts:
        """Write forecast CSV and manifest, return artifact metadata."""
        arts = ForecastArtifacts()

        if req.write_output and req.out_path is not None:
            t0 = time.monotonic()
            req.out_path.parent.mkdir(parents=True, exist_ok=True)
            result.forecast_df.to_csv(req.out_path)
            arts = ForecastArtifacts(
                forecast_csv=req.out_path,
                rows_written=len(result.forecast_df),
                duration_s=time.monotonic() - t0,
            )
            log.info("forecast csv  run_id=%s path=%s rows=%d", req.run_id, arts.forecast_csv, arts.rows_written)

        # Always write manifest
        cfg = req.config or FullRLMConfig(symbol=req.symbol)
        manifest = RunManifest(
            run_id=req.run_id,
            command="forecast",
            symbol=req.symbol,
            timestamp_utc=now_utc(),
            config_summary=build_config_summary(cfg),
            input_paths={"bars": str(req.bars_df.index[0]) + "…" if not req.bars_df.empty else ""},
            output_paths={"forecast_csv": str(arts.forecast_csv) if arts.forecast_csv else ""},
            metrics=self.summarize(result),
            backend=req.backend,
            profile=req.profile,
            success=True,
            duration_s=arts.duration_s,
        )
        manifest.write(req.data_root)
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
            "size_fraction": last_policy.get("roee_size_fraction") if last_policy is not None else None,
        }
        if result.backtest_metrics:
            summary["backtest"] = result.backtest_metrics
        return summary

    @staticmethod
    def output_columns(result: PipelineResult) -> list[str]:
        """Return the subset of standard output columns present in the result."""
        return [c for c in _OUTPUT_COLUMNS if c in result.forecast_df.columns]
