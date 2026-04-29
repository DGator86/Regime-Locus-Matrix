"""TradeService — decision generation and broker execution orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rlm.core.config import build_pipeline_config
from rlm.core.pipeline import FullRLMPipeline
from rlm.core.run_manifest import RunManifest, write_run_manifest
from rlm.data.paths import get_artifacts_dir
from rlm.data.readers import load_bars, load_option_chain
from rlm.execution.brokers import BrokerAdapter, IBKRBrokerAdapter
from rlm.utils.logging import get_run_logger
from rlm.utils.run_id import generate_run_id


@dataclass
class TradeRequest:
    """Input bundle for a trade plan / execution run."""

    symbol: str
    mode: str = "plan"
    use_kronos: bool = True
    attach_vix: bool = True
    capital: float = 100_000.0
    bars_df: pd.DataFrame | None = None
    option_chain_df: pd.DataFrame | None = None
    data_root: str | None = None
    backend: str = "auto"
    profile: str | None = None
    config_path: str | None = None
    out_dir: Path | None = None
    write_artifacts: bool = True
    use_persona: bool = False
    """When True, run the four-stage persona interpretation layer and attach the
    result to ``TradeResult.persona``."""


@dataclass
class TradeDecision:
    action: str | None
    strategy: str | None
    size_fraction: float | None
    vault_triggered: bool | None
    raw: dict[str, Any]


@dataclass
class TradeExecutionRecord:
    success: bool
    broker: str
    order_id: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeArtifacts:
    decision_path: Path | None = None
    execution_path: Path | None = None
    manifest_path: Path | None = None


@dataclass
class TradeResult:
    decision: TradeDecision
    executions: list[TradeExecutionRecord]
    artifacts: TradeArtifacts
    duration_s: float
    run_id: str
    persona: Any = None
    """PersonaPipelineResult when ``TradeRequest.use_persona`` is True; else None."""


class TradeService:
    """Generate a trade decision and optionally execute through a broker adapter."""

    def __init__(self, broker: BrokerAdapter | None = None) -> None:
        self._broker = broker or IBKRBrokerAdapter()

    def run(self, req: TradeRequest) -> TradeResult:
        t0 = time.monotonic()
        run_id = generate_run_id("trade")
        logger = get_run_logger(
            __name__,
            run_id,
            command="trade",
            symbol=req.symbol,
            backend=req.backend,
            profile=req.profile,
        )
        logger.info("building trade decision")
        decision, pipeline_result = self.build_decision(req, return_pipeline_result=True)
        logger.info("executing trade decision", extra={"stage": "execution"})
        executions = self.execute_decision(req, decision)
        logger.info("writing trade artifacts", extra={"stage": "artifacts"})
        artifacts = self.write_outputs(req, decision, executions, run_id)

        persona = None
        if req.use_persona:
            from rlm.persona.pipeline import PersonaDecisionPipeline

            logger.info("running persona interpretation layer", extra={"stage": "persona"})
            persona = PersonaDecisionPipeline().run(pipeline_result)

        return TradeResult(
            decision=decision,
            executions=executions,
            artifacts=artifacts,
            duration_s=time.monotonic() - t0,
            run_id=run_id,
            persona=persona,
        )

    def build_decision(
        self,
        req: TradeRequest,
        return_pipeline_result: bool = False,
    ) -> TradeDecision | tuple[TradeDecision, Any]:
        bars_df = (
            req.bars_df
            if req.bars_df is not None
            else load_bars(req.symbol, data_root=req.data_root, backend=req.backend)
        )
        chain_df = (
            req.option_chain_df
            if req.option_chain_df is not None
            else load_option_chain(req.symbol, data_root=req.data_root, backend=req.backend)
        )

        cfg = build_pipeline_config(
            symbol=req.symbol,
            profile=req.profile,
            config_path=req.config_path,
            overrides={
                "use_kronos": req.use_kronos,
                "attach_vix": req.attach_vix,
                "initial_capital": req.capital,
            },
        )
        result = FullRLMPipeline(cfg).run(bars_df, chain_df)

        if result.policy_df.empty:
            decision = TradeDecision(None, None, None, None, {"reason": "no_policy_rows"})
        else:
            last_row = result.policy_df.iloc[-1].to_dict()
            decision = TradeDecision(
                action=last_row.get("roee_action"),
                strategy=last_row.get("roee_strategy"),
                size_fraction=last_row.get("roee_size_fraction"),
                vault_triggered=last_row.get("vault_triggered"),
                raw=last_row,
            )

        if return_pipeline_result:
            return decision, result
        return decision

    def execute_decision(self, req: TradeRequest, decision: TradeDecision) -> list[TradeExecutionRecord]:
        if req.mode == "plan":
            return [
                TradeExecutionRecord(
                    success=True,
                    broker="none",
                    order_id=None,
                    message="Plan mode — no orders placed.",
                    details={"action": decision.action},
                )
            ]

        payload = {
            "roee_action": decision.action,
            "roee_strategy": decision.strategy,
            "roee_size_fraction": decision.size_fraction,
            "vault_triggered": decision.vault_triggered,
            **decision.raw,
        }
        try:
            rsp = self._broker.submit_trade_decision(req.symbol, payload, paper=(req.mode == "paper"))
            records = [
                TradeExecutionRecord(
                    success=bool(rsp.get("success")),
                    broker=str(rsp.get("broker") or "unknown"),
                    order_id=None if rsp.get("order_id") is None else str(rsp.get("order_id")),
                    message=str(rsp.get("message", "")),
                    details=dict(rsp.get("details") or {}),
                )
            ]
        except Exception as exc:
            records = [
                TradeExecutionRecord(
                    success=False,
                    broker=getattr(self._broker, "broker", "unknown"),
                    order_id=None,
                    message=str(exc),
                    details={"mode": req.mode},
                )
            ]
        return records or [
            TradeExecutionRecord(
                success=False,
                broker="unknown",
                order_id=None,
                message="No execution record produced.",
                details={},
            )
        ]

    def write_outputs(
        self,
        req: TradeRequest,
        decision: TradeDecision,
        executions: list[TradeExecutionRecord],
        run_id: str,
    ) -> TradeArtifacts:
        if not req.write_artifacts:
            return TradeArtifacts()

        out_dir = req.out_dir or (get_artifacts_dir(req.data_root) / "trade" / run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        decision_path = out_dir / "decision.json"
        execution_path = out_dir / "execution.json"
        manifest_path = out_dir / "run_manifest.json"

        decision_path.write_text(
            json.dumps(
                {
                    "symbol": req.symbol,
                    "action": decision.action,
                    "strategy": decision.strategy,
                    "size_fraction": decision.size_fraction,
                    "vault_triggered": decision.vault_triggered,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        execution_path.write_text(
            json.dumps(
                [
                    {
                        "success": x.success,
                        "broker": x.broker,
                        "order_id": x.order_id,
                        "message": x.message,
                        "details": x.details,
                    }
                    for x in executions
                ],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        manifest_path = write_run_manifest(
            RunManifest(
                run_id=run_id,
                command="trade",
                symbol=req.symbol,
                timestamp_utc=datetime.now(tz=UTC).isoformat(),
                backend=req.backend,
                profile=req.profile,
                config_summary={
                    "config_path": req.config_path,
                    "use_kronos": req.use_kronos,
                    "attach_vix": req.attach_vix,
                    "capital": req.capital,
                },
                input_paths={},
                output_paths={
                    "decision_path": str(decision_path),
                    "execution_path": str(execution_path),
                    "manifest_path": str(manifest_path),
                },
                metrics={
                    "executions_total": len(executions),
                    "executions_success": sum(1 for x in executions if x.success),
                },
            ),
            data_root=req.data_root,
            out_path=manifest_path,
        )
        return TradeArtifacts(decision_path=decision_path, execution_path=execution_path, manifest_path=manifest_path)

    def summarize(self, result: TradeResult) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "action": result.decision.action,
            "strategy": result.decision.strategy,
            "executions": len(result.executions),
            "success": all(x.success for x in result.executions) if result.executions else True,
            "duration_s": result.duration_s,
        }
