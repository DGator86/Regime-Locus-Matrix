"""TradeService — application layer for live/paper trade plan generation and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.core.run_manifest import RunManifest, build_config_summary, new_run_id, now_utc
from rlm.data.readers import load_bars, load_option_chain
from rlm.utils.logging import get_logger
from rlm.utils.timing import timed_stage

log = get_logger(__name__)


@dataclass
class TradeRequest:
    """Input bundle for a trade plan / execution run."""

    symbol: str
    mode: str = "plan"  # "plan" | "paper" | "live"
    use_kronos: bool = True
    attach_vix: bool = True
    capital: float = 100_000.0
    bars_df: pd.DataFrame | None = None
    option_chain_df: pd.DataFrame | None = None
    config: FullRLMConfig | None = None
    data_root: str | None = None
    backend: str = "auto"
    profile: str | None = None
    write_output: bool = False
    out_dir: Path | None = None
    run_id: str = field(default_factory=new_run_id)


@dataclass
class TradeArtifacts:
    """Paths and metadata for files written by TradeService."""

    decision_json: Path | None = None
    duration_s: float = 0.0


@dataclass
class TradeResult:
    decision: dict | None = None
    execution_log: list[str] = field(default_factory=list)
    pipeline_result: PipelineResult | None = None
    duration_s: float = 0.0


class TradeService:
    """Generate a trade decision from the latest market data and optionally execute it.

    In ``plan`` mode the decision is printed only — no orders are placed.
    In ``paper`` / ``live`` mode the decision is forwarded to the IBKR execution layer.

    Data loading uses ``rlm.data.readers`` so no ``__file__``-relative paths appear here.
    """

    def run(self, req: TradeRequest) -> TradeResult:
        cfg = req.config or FullRLMConfig(
            symbol=req.symbol,
            use_kronos=req.use_kronos,
            attach_vix=req.attach_vix,
            initial_capital=req.capital,
        )

        with timed_stage(
            log, "trade",
            run_id=req.run_id, symbol=req.symbol,
            mode=req.mode, kronos=cfg.use_kronos,
        ):
            bars_df = req.bars_df
            chain_df = req.option_chain_df

            if bars_df is None:
                bars_df = load_bars(req.symbol, data_root=req.data_root, backend=req.backend)
            if chain_df is None:
                chain_df = load_option_chain(req.symbol, data_root=req.data_root, backend=req.backend)

            pipeline_result = FullRLMPipeline(cfg).run(bars_df, chain_df)

            if pipeline_result.policy_df.empty:
                log.warning("trade no policy rows  symbol=%s", req.symbol)
                return TradeResult(
                    execution_log=["No policy rows — cannot generate decision."],
                    pipeline_result=pipeline_result,
                )

            last_row = pipeline_result.policy_df.iloc[-1].to_dict()
            decision = {
                "roee_action": last_row.get("roee_action"),
                "roee_strategy": last_row.get("roee_strategy"),
                "roee_size_fraction": last_row.get("roee_size_fraction"),
                "vault_triggered": last_row.get("vault_triggered"),
            }

            log.info(
                "trade decision  symbol=%s action=%s strategy=%s size=%s",
                req.symbol,
                decision["roee_action"],
                decision["roee_strategy"],
                decision["roee_size_fraction"],
            )

            if req.mode in ("paper", "live"):
                execution_log = self._execute(req, decision)
            else:
                execution_log = ["Plan mode — no orders placed."]

        return TradeResult(
            decision=decision,
            execution_log=execution_log,
            pipeline_result=pipeline_result,
        )

    def write_outputs(self, req: TradeRequest, result: TradeResult) -> TradeArtifacts:
        """Write decision JSON and manifest.  Returns artifact metadata."""
        import json
        import time

        arts = TradeArtifacts()
        t0 = time.monotonic()

        if req.write_output and req.out_dir is not None and result.decision is not None:
            req.out_dir.mkdir(parents=True, exist_ok=True)
            decision_path = req.out_dir / f"trade_decision_{req.symbol}.json"
            decision_path.write_text(
                json.dumps(
                    {"run_id": req.run_id, "decision": result.decision, "log": result.execution_log},
                    indent=2,
                ),
                encoding="utf-8",
            )
            arts.decision_json = decision_path
            log.info("trade decision written  path=%s", decision_path)

        arts.duration_s = time.monotonic() - t0

        cfg = req.config or FullRLMConfig(symbol=req.symbol)
        manifest = RunManifest(
            run_id=req.run_id,
            command="trade",
            symbol=req.symbol,
            timestamp_utc=now_utc(),
            config_summary=build_config_summary(cfg),
            output_paths={
                "decision_json": str(arts.decision_json) if arts.decision_json else "",
            },
            metrics=self.summarize(result),
            backend=req.backend,
            profile=req.profile,
            success=True,
            duration_s=arts.duration_s,
        )
        manifest.write(req.data_root)
        return arts

    def summarize(self, result: TradeResult) -> dict:
        """Return a human-readable summary dict from the trade result."""
        summary: dict = {
            "mode": "unknown",
            "execution_log": result.execution_log,
        }
        if result.decision:
            summary.update(result.decision)
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(self, req: TradeRequest, decision: dict) -> list[str]:
        action = decision.get("roee_action", "hold")
        if action == "hold":
            return ["Action=hold — no orders submitted."]

        try:
            from rlm.execution.ibkr_combo import place_roee_combo

            log.info("ibkr order  symbol=%s action=%s mode=%s", req.symbol, action, req.mode)
            order_id = place_roee_combo(
                symbol=req.symbol,
                decision=decision,
                paper=(req.mode == "paper"),
            )
            return [f"Order placed: id={order_id} action={action}"]
        except Exception as exc:
            log.error("ibkr order failed  symbol=%s error=%s", req.symbol, exc)
            return [f"Execution failed: {exc}"]
