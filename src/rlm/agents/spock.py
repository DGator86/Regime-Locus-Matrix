"""
Spock — logical market analyst.

"Fascinating. The probability of a profitable outcome, Captain, is precisely
47.3%… assuming the regime signal is correct."

Reads current trade plans, regime signals, and Kronos forecast artefacts,
then synthesises a concise, probability-weighted analysis of each active plan.
Outputs a structured recommendation list for Kirk to act on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rlm.agents.base import LLMClient, Message
from rlm.persona.pipeline import PersonaDecisionPipeline
from rlm.persona.models import PersonaInputs

# -----------------------------------------------------------------------
_SPOCK_SYSTEM = """\
You are Spock, the Science Officer and strategic analyst of this trading system.
You operate on pure logic and probability. No emotional language. No hedging phrases.
Structure your response as a numbered list of active trade plans:
  1. SYMBOL | STRATEGY | REGIME | ACTION: [GO / HOLD / ABORT] | RATIONALE: <one sentence, cite numbers>
End with a single LINE: OVERALL RISK POSTURE: [LOW / MODERATE / HIGH / CRITICAL]
Keep each entry under 25 words. Do not repeat data already provided verbatim.
"""


@dataclass
class PlanAnalysis:
    symbol: str
    plan_id: str
    strategy: str
    regime: str
    rank_score: float
    action: str          # GO | HOLD | ABORT
    rationale: str
    persona_directive: str = "unknown"
    persona_trap_detected: bool = False
    raw_plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpockBriefing:
    timestamp: str
    plan_analyses: list[PlanAnalysis] = field(default_factory=list)
    overall_risk: str = "UNKNOWN"
    llm_text: str = ""
    context_snapshot: str = ""


class SpockAgent:
    def __init__(self, root: Path, llm: Optional[LLMClient] = None) -> None:
        self.root = root
        self.llm = llm or LLMClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self) -> SpockBriefing:
        """Read current artefacts and return Spock's briefing."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        briefing = SpockBriefing(timestamp=ts)

        context = self._build_context()
        briefing.context_snapshot = context

        if not context.strip() or "no active plans" in context.lower():
            briefing.llm_text = "No active trade plans to analyse at this time. Standing by."
            briefing.overall_risk = "LOW"
            return briefing

        try:
            briefing.llm_text = self.llm.chat(
                [Message("user",
                    f"Analyse these active trade plans and regime signals:\n\n{context}\n\n"
                    "Provide your logical assessment per the format in your instructions.")],
                system=_SPOCK_SYSTEM,
            )
            briefing.overall_risk = self._extract_risk(briefing.llm_text)
        except Exception as exc:
            briefing.llm_text = f"[Spock LLM unavailable: {exc}]\n\n{context}"

        return briefing

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        sections: list[str] = []

        # 1 — Universe trade plans
        plans_text = self._read_plans()
        if plans_text:
            sections.append(f"=== Active Trade Plans ===\n{plans_text}")

        # 2 — Latest run artefact (regime + Kronos signals)
        artifact_text = self._read_latest_artifact()
        if artifact_text:
            sections.append(f"=== Latest Pipeline Signals ===\n{artifact_text}")

        # 3 — Equity positions
        eq_text = self._read_equity_state()
        if eq_text:
            sections.append(f"=== Open Equity Positions ===\n{eq_text}")

        return "\n\n".join(sections) if sections else "No active plans or signals found."

    def _read_plans(self) -> str:
        path = self.root / "data" / "processed" / "universe_trade_plans.json"
        if not path.is_file():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        results = data.get("results") or []
        active = [r for r in results if r.get("status") == "active"]
        if not active:
            return "No active plans."
        lines: list[str] = [
            f"Generated: {data.get('generated_at_utc', '?')}",
            f"Active plans: {len(active)}",
        ]
        active.sort(key=lambda x: float(x.get("rank_score") or 0), reverse=True)
        for r in active[:15]:
            sym = r.get("symbol", "?")
            strat = r.get("strategy", "?")
            score = r.get("rank_score", "?")
            pid = r.get("plan_id", "?")
            regime = r.get("regime", r.get("regime_label", "?"))
            conf = r.get("regime_confidence", r.get("confidence", "?"))
            kron = r.get("kronos_return_forecast", "?")

            # Run persona interpretation if possible
            persona_str = ""
            try:
                pipeline_data = r.get("pipeline", {})
                regime_key = str(pipeline_data.get("regime_key", ""))
                regime_parts = [p.strip() for p in regime_key.split("|") if p.strip()]
                
                inputs = PersonaInputs(
                    s_d=float(pipeline_data.get("S_D", 0.0)),
                    s_v=float(pipeline_data.get("S_V", 0.0)),
                    s_l=float(pipeline_data.get("S_L", 0.0)),
                    s_g=float(pipeline_data.get("S_G", 0.0)),
                    direction_regime=regime_parts[0] if len(regime_parts) > 0 else "neutral",
                    volatility_regime=regime_parts[1] if len(regime_parts) > 1 else "neutral",
                    liquidity_regime=regime_parts[2] if len(regime_parts) > 2 else "neutral",
                    dealer_flow_regime=regime_parts[3] if len(regime_parts) > 3 else "neutral",
                    hmm_confidence=float(r.get("decision", {}).get("metadata", {}).get("confidence", 0.5)),
                    roee_action=str(r.get("decision", {}).get("strategy_name", "")) if r.get("decision") else None,
                )
                persona = PersonaDecisionPipeline().run_from_inputs(inputs)
                trap = " [TRAP!]" if persona.garak.trap_detected else ""
                persona_str = f" | SISKO: {persona.sisko.directive.upper()}{trap}"
            except Exception:
                pass

            lines.append(
                f"  • {sym} [{strat}] score={score} regime={regime} conf={conf} "
                f"kronos={kron}{persona_str} id={pid}"
            )
        return "\n".join(lines)

    def _read_latest_artifact(self) -> str:
        artifacts_dir = self.root / "data" / "artifacts" / "runs"
        if not artifacts_dir.is_dir():
            return ""
        try:
            files = sorted(artifacts_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not files:
                return ""
            latest = json.loads(files[0].read_text(encoding="utf-8"))
            lines: list[str] = [f"Run: {files[0].name}"]
            for key in ("regime", "regime_label", "regime_confidence",
                        "kronos_return_forecast", "kronos_confidence",
                        "kronos_regime_agreement", "kronos_transition_flag",
                        "roee_strategy", "roee_confidence"):
                val = latest.get(key)
                if val is not None:
                    lines.append(f"  {key}: {val}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception:
            return ""

    def _read_equity_state(self) -> str:
        path = self.root / "data" / "processed" / "equity_positions_state.json"
        if not path.is_file():
            return ""
        try:
            data: dict = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        open_pos = [(k, v) for k, v in data.items() if str((v or {}).get("status")) == "open"]
        if not open_pos:
            return "No open equity positions."
        lines = []
        for pid, pos in open_pos[:10]:
            sym = (pos or {}).get("symbol", "?")
            side = (pos or {}).get("side", "?")
            qty = (pos or {}).get("quantity", "?")
            lines.append(f"  • {sym} {side} qty={qty} id={pid}")
        return "\n".join(lines)

    @staticmethod
    def _extract_risk(text: str) -> str:
        for line in reversed(text.splitlines()):
            upper = line.upper()
            for level in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
                if level in upper and "RISK POSTURE" in upper:
                    return level
        return "UNKNOWN"
