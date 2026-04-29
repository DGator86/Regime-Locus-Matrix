"""BacktestService — application layer for backtesting and walk-forward runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from rlm.core.config import build_pipeline_config
from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.data.paths import get_processed_data_dir
from rlm.datasets.paths import (
    walkforward_equity_filename,
    walkforward_summary_filename,
    walkforward_trades_filename,
)
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
        cfg = req.config or build_pipeline_config(symbol=req.symbol, overrides={})
        t0 = time.monotonic()

        # Guarantee backtest is on, and apply capital override if supplied
        overrides: dict = {"run_backtest": True}
        if req.initial_capital is not None:
            overrides["initial_capital"] = req.initial_capital
        cfg = replace(cfg, **overrides)

        log.info(
            "backtest start  symbol=%s regime=%s kronos=%s walkforward=%s bars=%d capital=%.0f",
            req.symbol,
            cfg.regime_model,
            cfg.use_kronos,
            req.walkforward,
            len(req.bars_df),
            cfg.initial_capital,
        )

        with timed_stage(log, "backtest_pipeline", symbol=req.symbol):
            result = FullRLMPipeline(cfg).run(req.bars_df, req.option_chain_df)

        if req.walkforward and result.policy_df is not None:
            result = self._run_walkforward(req, cfg, result)

        log.info(
            "backtest done  symbol=%s duration=%.1fs metrics=%s",
            req.symbol,
            time.monotonic() - t0,
            result.backtest_metrics,
        )
        return result

    def write_outputs(self, req: BacktestRequest, result: PipelineResult) -> BacktestArtifacts:
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
        return arts

    def summarize(self, result: PipelineResult) -> dict:
        """Return a human-readable summary dict from the backtest result."""
        summary: dict = {}
        if result.backtest_metrics:
            summary["metrics"] = result.backtest_metrics
        if result.backtest_trades is not None:
            summary["trade_count"] = len(result.backtest_trades)
        if result.walkforward_summary is not None and not result.walkforward_summary.empty:
            summary["walkforward_windows"] = len(result.walkforward_summary)
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_walkforward(self, req: BacktestRequest, cfg: FullRLMConfig, base_result: PipelineResult) -> PipelineResult:
        from rlm.backtest.walkforward import WalkForwardConfig, WalkForwardEngine

        out_dir = req.out_dir if req.out_dir is not None else get_processed_data_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        hmm_dir = out_dir / "walkforward_hmm"
        hmm_dir.mkdir(parents=True, exist_ok=True)
        try:
            engine = WalkForwardEngine(
                pipeline_config=cfg,
                wf_config=WalkForwardConfig(),
                hmm_model_dir=hmm_dir,
            )
            wf_result = engine.run(req.bars_df, req.option_chain_df)
            log.info(
                "walkforward done  symbol=%s windows=%d trades=%d",
                req.symbol,
                len(wf_result.summary_df),
                len(wf_result.trades_df),
            )
        except Exception as exc:
            log.warning("walkforward failed  symbol=%s error=%s", req.symbol, exc)
            return base_result

        if req.write_outputs:
            sym = req.symbol.upper()
            sum_path = out_dir / walkforward_summary_filename(sym)
            wf_result.summary_df.to_csv(sum_path, index=False)
            if not wf_result.equity_df.empty:
                wf_result.equity_df.to_csv(out_dir / walkforward_equity_filename(sym))
            if not wf_result.trades_df.empty:
                wf_result.trades_df.to_csv(out_dir / walkforward_trades_filename(sym))
            log.info("walkforward output  summary=%s", sum_path)

        merged_metrics = dict(base_result.backtest_metrics or {})
        merged_metrics.update(_aggregate_walkforward_metrics(wf_result.summary_df))
        return replace(
            base_result,
            backtest_metrics=merged_metrics,
            walkforward_summary=wf_result.summary_df,
        )


def _aggregate_walkforward_metrics(summary_df: pd.DataFrame) -> dict[str, float]:
    if summary_df.empty:
        return {}
    out: dict[str, float] = {"wf_windows": float(len(summary_df))}
    for col in (
        "sharpe",
        "total_return_pct",
        "max_drawdown",
        "num_trades",
        "win_rate",
        "profit_factor",
        "final_equity",
    ):
        if col in summary_df.columns:
            s = pd.to_numeric(summary_df[col], errors="coerce")
            out[f"wf_mean_{col}"] = float(s.mean())
    return out
