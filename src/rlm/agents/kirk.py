"""
Kirk — strategy commander and decision-maker.

"Bones, I need options. Spock, what are the odds? Scotty, keep us in one piece."

Kirk receives health reports from Scotty and market analysis from Spock,
then synthesises a clear executive command decision: proceed, stand down,
or escalate to the human operator via Telegram.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rlm.agents.base import LLMClient, Message
from rlm.agents.scotty import HealthReport
from rlm.agents.spock import SpockBriefing

# -----------------------------------------------------------------------
_KIRK_SYSTEM = """\
You are Captain Kirk, the commanding officer of this trading system.
You have received reports from your Chief Engineer (Scotty), Science Officer (Spock), 
and the tactical interpretation team (Sisko, Garak, and Seven).

Your role: make the final command decision and communicate it clearly to the crew.

The Sisko team provides technical advisories:
- Seven: Signal normalization and bias interpretation.
- Garak: Trap and deception detection (veto logic).
- Sisko: Final trade directive (long / short / no_trade).

Response format (plain text, no markdown):
SYSTEM STATUS: [NOMINAL / DEGRADED / CRITICAL]
MARKET POSTURE: [AGGRESSIVE / NORMAL / DEFENSIVE / STAND-DOWN]
COMMAND DECISION: <one decisive sentence — GO / HOLD / STAND-DOWN / ALERT OPERATOR>
RATIONALE: <2-3 sentences max, referencing Scotty, Spock, and the Sisko team's findings>
CREW ORDERS:
  - Scotty: <one action item or "maintain current status">
  - Spock: <one action item or "continue monitoring">
  - Sisko Team: <one directive for tactical interpreters>
  - Helm: <one directive for the trading engine>
"""

_RISK_TO_POSTURE = {
    "LOW": "AGGRESSIVE",
    "MODERATE": "NORMAL",
    "HIGH": "DEFENSIVE",
    "CRITICAL": "STAND-DOWN",
    "UNKNOWN": "NORMAL",
}


@dataclass
class CommandDecision:
    timestamp: str
    system_status: str        # NOMINAL | DEGRADED | CRITICAL
    market_posture: str       # AGGRESSIVE | NORMAL | DEFENSIVE | STAND-DOWN
    command: str              # GO | HOLD | STAND-DOWN | ALERT OPERATOR
    rationale: str
    crew_orders: dict[str, str] = field(default_factory=dict)
    llm_text: str = ""
    needs_human: bool = False

    def to_telegram_message(self) -> str:
        orders = "\n".join(f"  {k}: {v}" for k, v in self.crew_orders.items())
        flag = " ** HUMAN REVIEW REQUESTED **" if self.needs_human else ""
        return (
            f"[Kirk]{flag}\n"
            f"System: {self.system_status} | Market: {self.market_posture}\n"
            f"Decision: {self.command}\n"
            f"Rationale: {self.rationale}\n"
            f"Crew orders:\n{orders}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_status": self.system_status,
            "market_posture": self.market_posture,
            "command": self.command,
            "rationale": self.rationale,
            "crew_orders": self.crew_orders,
            "needs_human": self.needs_human,
        }


class KirkAgent:
    def __init__(
        self,
        root: Path,
        llm: Optional[LLMClient] = None,
        *,
        alert_on_degraded: bool = True,
        alert_on_high_risk: bool = True,
    ) -> None:
        self.root = root
        self.llm = llm or LLMClient()
        self.alert_on_degraded = alert_on_degraded
        self.alert_on_high_risk = alert_on_high_risk
        self._decisions_path = root / "data" / "artifacts" / "crew_decisions.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def command(
        self,
        health: HealthReport,
        briefing: SpockBriefing,
        scotty_diagnosis: str,
    ) -> CommandDecision:
        """Synthesise Scotty + Spock reports and issue a command decision."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        context = self._build_context(health, scotty_diagnosis, briefing)

        try:
            llm_text = self.llm.chat(
                [Message("user",
                    f"Here are your crew reports:\n\n{context}\n\n"
                    "Issue your command decision.")],
                system=_KIRK_SYSTEM,
            )
        except Exception as exc:
            llm_text = f"[Kirk LLM unavailable: {exc}]\nDefaulting to HOLD."

        decision = self._parse(ts, llm_text, health, briefing)
        self._save(decision)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        health: HealthReport,
        scotty_diagnosis: str,
        briefing: SpockBriefing,
    ) -> str:
        parts = [
            "=== Scotty's Engineering Report ===",
            health.to_text(),
            "",
            "=== Scotty's Diagnosis ===",
            scotty_diagnosis,
            "",
            "=== Spock's Market Briefing ===",
            briefing.llm_text or briefing.context_snapshot,
        ]
        return "\n".join(parts)

    def _parse(
        self,
        ts: str,
        text: str,
        health: HealthReport,
        briefing: SpockBriefing,
    ) -> CommandDecision:
        # Derive fallback values from structured data
        sys_status = "NOMINAL" if health.overall_ok else "DEGRADED"
        mkt_posture = _RISK_TO_POSTURE.get(briefing.overall_risk, "NORMAL")
        command_str = "HOLD"
        rationale = "Defaulted due to LLM parse failure."
        orders: dict[str, str] = {}

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("SYSTEM STATUS:"):
                val = stripped.split(":", 1)[1].strip()
                if val in ("NOMINAL", "DEGRADED", "CRITICAL"):
                    sys_status = val
            elif stripped.startswith("MARKET POSTURE:"):
                val = stripped.split(":", 1)[1].strip()
                if val in ("AGGRESSIVE", "NORMAL", "DEFENSIVE", "STAND-DOWN"):
                    mkt_posture = val
            elif stripped.startswith("COMMAND DECISION:"):
                command_str = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("RATIONALE:"):
                rationale = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- ") and ":" in stripped:
                key, _, val = stripped[2:].partition(":")
                orders[key.strip()] = val.strip()

        needs_human = (
            (self.alert_on_degraded and not health.overall_ok)
            or (self.alert_on_high_risk and briefing.overall_risk in ("HIGH", "CRITICAL"))
            or "ALERT OPERATOR" in command_str.upper()
        )

        return CommandDecision(
            timestamp=ts,
            system_status=sys_status,
            market_posture=mkt_posture,
            command=command_str,
            rationale=rationale,
            crew_orders=orders,
            llm_text=text,
            needs_human=needs_human,
        )

    def _save(self, decision: CommandDecision) -> None:
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing: list = []
            if self._decisions_path.is_file():
                existing = json.loads(self._decisions_path.read_text(encoding="utf-8"))
            existing.append(decision.to_json())
            existing = existing[-500:]  # keep last 500 decisions
            self._decisions_path.write_text(
                json.dumps(existing, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
