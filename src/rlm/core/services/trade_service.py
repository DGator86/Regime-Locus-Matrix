"""TradeService — decision generation and broker execution orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from rlm.cli.common import build_pipeline_config_from_controls
from rlm.core.pipeline import FullRLMPipeline
from rlm.data.paths import get_artifacts_dir
from rlm.data.readers import load_bars, load_option_chain
from rlm.execution.brokers import BrokerAdapter, IBKRBrokerAdapter
from rlm.utils.logging import get_logger

log = get_logger(__name__)


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


@dataclass
class TradeDecision:
    action: str | None
    strategy: str | None
    size_fraction: float | None
    vault_triggered: bool | None
    raw: dict[str, Any]


@dataclass
class TradeExecutionRecord:
    mode: str
    success: bool
    broker: str | None = None
    order_id: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeArtifacts:
    decision_json: Path | None = None
    execution_json: Path | None = None
    manifest_path: Path | None = None


@dataclass
class TradeResult:
    decision: TradeDecision | None = None
    executions: list[TradeExecutionRecord] = field(default_factory=list)
    artifacts: TradeArtifacts = field(default_factory=TradeArtifacts)
    duration_s: float = 0.0
    run_id: str | None = None


class TradeService:
    """Generate a trade decision and optionally execute through a broker adapter."""

    def __init__(self, broker: BrokerAdapter | None = None) -> None:
        self._broker = broker or IBKRBrokerAdapter()

    def run(self, req: TradeRequest) -> TradeResult:
        t0 = time.monotonic()
        run_id = self._new_run_id(req.symbol)

        decision = self.build_decision(req)
        executions = self.execute_decision(req, decision)
        result = TradeResult(decision=decision, executions=executions, run_id=run_id)

        if req.write_artifacts:
            result.artifacts = self.write_outputs(req, result)

        result.duration_s = time.monotonic() - t0
        return result

    def build_decision(self, req: TradeRequest) -> TradeDecision:
        bars_df = req.bars_df if req.bars_df is not None else load_bars(req.symbol, data_root=req.data_root, backend=req.backend)
        chain_df = req.option_chain_df if req.option_chain_df is not None else load_option_chain(req.symbol, data_root=req.data_root, backend=req.backend)

        cfg = build_pipeline_config_from_controls(
            symbol=req.symbol,
            profile=req.profile,
            config_path=req.config_path,
            use_kronos=req.use_kronos,
            attach_vix=req.attach_vix,
            initial_capital=req.capital,
        )
        result = FullRLMPipeline(cfg).run(bars_df, chain_df)

        if result.policy_df.empty:
            return TradeDecision(None, None, None, None, {"reason": "no_policy_rows"})

        last_row = result.policy_df.iloc[-1].to_dict()
        return TradeDecision(
            action=last_row.get("roee_action"),
            strategy=last_row.get("roee_strategy"),
            size_fraction=last_row.get("roee_size_fraction"),
            vault_triggered=last_row.get("vault_triggered"),
            raw=last_row,
        )

    def execute_decision(self, req: TradeRequest, decision: TradeDecision) -> list[TradeExecutionRecord]:
        if req.mode == "plan":
            return [
                TradeExecutionRecord(
                    mode=req.mode,
                    success=True,
                    broker=None,
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
        rsp = self._broker.submit_trade_decision(req.symbol, payload, paper=(req.mode == "paper"))
        return [
            TradeExecutionRecord(
                mode=req.mode,
                success=bool(rsp.get("success")),
                broker=rsp.get("broker"),
                order_id=rsp.get("order_id"),
                message=str(rsp.get("message", "")),
                details=dict(rsp.get("details") or {}),
            )
        ]

    def write_outputs(self, req: TradeRequest, result: TradeResult) -> TradeArtifacts:
        out_dir = req.out_dir or (get_artifacts_dir(req.data_root) / "trade" / str(result.run_id))
        out_dir.mkdir(parents=True, exist_ok=True)

        decision_json = out_dir / "decision.json"
        execution_json = out_dir / "execution.json"
        manifest_path = out_dir / "manifest.json"

        decision_payload = {
            "run_id": result.run_id,
            "symbol": req.symbol,
            "mode": req.mode,
            "decision": None
            if result.decision is None
            else {
                "action": result.decision.action,
                "strategy": result.decision.strategy,
                "size_fraction": result.decision.size_fraction,
                "vault_triggered": result.decision.vault_triggered,
                "raw": result.decision.raw,
            },
        }
        decision_json.write_text(json.dumps(decision_payload, indent=2, default=str), encoding="utf-8")

        execution_payload = {
            "run_id": result.run_id,
            "mode": req.mode,
            "executions": [
                {
                    "mode": x.mode,
                    "success": x.success,
                    "broker": x.broker,
                    "order_id": x.order_id,
                    "message": x.message,
                    "details": x.details,
                }
                for x in result.executions
            ],
        }
        execution_json.write_text(json.dumps(execution_payload, indent=2, default=str), encoding="utf-8")

        manifest_payload = {
            "run_id": result.run_id,
            "command": "trade",
            "symbol": req.symbol,
            "backend": req.backend,
            "profile": req.profile,
            "config_path": req.config_path,
            "artifacts": {
                "decision_json": str(decision_json),
                "execution_json": str(execution_json),
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        return TradeArtifacts(decision_json=decision_json, execution_json=execution_json, manifest_path=manifest_path)

    def summarize(self, result: TradeResult) -> dict[str, Any]:
        return {
            "run_id": result.run_id,
            "action": None if result.decision is None else result.decision.action,
            "strategy": None if result.decision is None else result.decision.strategy,
            "executions": len(result.executions),
            "success": all(x.success for x in result.executions) if result.executions else True,
        }

    @staticmethod
    def _new_run_id(symbol: str) -> str:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return f"{ts}_{symbol.upper()}_{uuid4().hex[:8]}"
