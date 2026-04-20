"""BacktestService — application layer for backtesting and walk-forward runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.core.run_manifest import RunManifest, build_config_summary, new_run_id, now_utc
from rlm.utils.logging import get_logger
from rlm.utils.timing import timed_stage

log = get_logger(__name__)


@dataclass
class BacktestRequest:
    """Input bundle for a single backtest run."""

    symbol: str
    bars_df: pd.DataFrame
    option_chain_df: pd.DataFrame | None = None
    config: FullRLMConfig | None = None
    walkforward: bool = False
    write_outputs: bool = True
    out_dir: Path | None = None
    initial_capital: float | None = None
    run_id: str = field(default_factory=new_run_id)
    data_root: str | None = None
    backend: str = "auto"
    profile: str | None = None


@dataclass
class BacktestArtifacts:
    """Paths and metadata for files written by BacktestService."""

    trades_csv: Path | None = None
    equity_csv: Path | None = None
    trades_rows: int = 0
    equity_rows: int = 0
    duration_s: float = 0.0


class BacktestService:
    """Orchestrate a full pipeline run followed by backtesting.

    Wraps ``FullRLMPipeline`` with ``run_backtest=True``, optionally chains a
    walk-forward validation pass, and owns all output file writing.
    """

    def run(self, req: BacktestRequest) -> PipelineResult:
        cfg = req.config or FullRLMConfig(symbol=req.symbol)

        # Guarantee backtest is on, and apply capital override if supplied
        overrides: dict = {"run_backtest": True}
        if req.initial_capital is not None:
            overrides["initial_capital"] = req.initial_capital
        cfg = replace(cfg, **overrides)

        with timed_stage(
            log, "backtest",
            run_id=req.run_id, symbol=req.symbol,
            regime=cfg.regime_model, walkforward=req.walkforward,
            capital=int(cfg.initial_capital),
        ):
            result = FullRLMPipeline(cfg).run(req.bars_df, req.option_chain_df)
            if req.walkforward and result.policy_df is not None:
                result = self._run_walkforward(req, cfg, result)

        return result

    def write_outputs(
        self, req: BacktestRequest, result: PipelineResult
    ) -> BacktestArtifacts:
        """Write trades and equity curve CSVs, return artifact metadata."""
        if not req.write_outputs or req.out_dir is None:
            return BacktestArtifacts()

        req.out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        arts = BacktestArtifacts(duration_s=0.0)

        if result.backtest_trades is not None and not result.backtest_trades.empty:
            arts.trades_csv = req.out_dir / f"backtest_trades_{req.symbol}.csv"
            result.backtest_trades.to_csv(arts.trades_csv)
            arts.trades_rows = len(result.backtest_trades)
            log.info("backtest output  trades=%s rows=%d", arts.trades_csv, arts.trades_rows)

        if result.backtest_equity is not None and not result.backtest_equity.empty:
            arts.equity_csv = req.out_dir / f"backtest_equity_{req.symbol}.csv"
            result.backtest_equity.to_csv(arts.equity_csv)
            arts.equity_rows = len(result.backtest_equity)
            log.info("backtest output  equity=%s rows=%d", arts.equity_csv, arts.equity_rows)

        arts.duration_s = time.monotonic() - t0

        # Write manifest
        cfg = req.config or FullRLMConfig(symbol=req.symbol)
        manifest = RunManifest(
            run_id=req.run_id,
            command="backtest",
            symbol=req.symbol,
            timestamp_utc=now_utc(),
            config_summary=build_config_summary(cfg),
            output_paths={
                "trades_csv": str(arts.trades_csv) if arts.trades_csv else "",
                "equity_csv": str(arts.equity_csv) if arts.equity_csv else "",
            },
            metrics=self.summarize(result),
            backend=req.backend,
            profile=req.profile,
            success=True,
            duration_s=arts.duration_s,
        )
        manifest.write(req.data_root)
        return arts

    def summarize(self, result: PipelineResult) -> dict:
        """Return a human-readable summary dict from the backtest result."""
        summary: dict = {}
        if result.backtest_metrics:
            summary["metrics"] = result.backtest_metrics
        if result.backtest_trades is not None:
            summary["trade_count"] = len(result.backtest_trades)
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_walkforward(
        self, req: BacktestRequest, cfg: FullRLMConfig, base_result: PipelineResult
    ) -> PipelineResult:
        try:
            from rlm.backtest.walkforward import WalkForwardEngine, WalkForwardConfig

            engine = WalkForwardEngine(pipeline_config=cfg, wf_config=WalkForwardConfig())
            wf_result = engine.run(req.bars_df, req.option_chain_df)
            log.info(
                "walkforward done  symbol=%s windows=%d",
                req.symbol, len(wf_result.window_results),
            )
        except Exception as exc:
            log.warning("walkforward failed  symbol=%s error=%s", req.symbol, exc)

        return base_result
