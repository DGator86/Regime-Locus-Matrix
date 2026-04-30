"""Parse commander-style LLM output and persist decisions (ex-Kirk)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RISK_TO_POSTURE = {
    "LOW": "AGGRESSIVE",
    "MODERATE": "NORMAL",
    "HIGH": "DEFENSIVE",
    "CRITICAL": "STAND-DOWN",
    "UNKNOWN": "NORMAL",
}

_RISK_PATTERN = re.compile(
    r"OVERALL\s+RISK\s+POSTURE\s*:\s*(CRITICAL|HIGH|MODERATE|LOW)",
    re.IGNORECASE,
)


@dataclass
class CommandDecision:
    timestamp: str
    system_status: str
    market_posture: str
    command: str
    rationale: str
    crew_orders: dict[str, str] = field(default_factory=dict)
    llm_text: str = ""
    needs_human: bool = False

    def to_telegram_message(self) -> str:
        orders = "\n".join(f"  {k}: {v}" for k, v in self.crew_orders.items())
        flag = " ** HUMAN REVIEW REQUESTED **" if self.needs_human else ""
        return (
            f"[Hermes Commander]{flag}\n"
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


def infer_overall_risk_from_text(text: str) -> str:
    for line in reversed(text.splitlines()):
        m = _RISK_PATTERN.search(line)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def parse_command_decision(
    ts: str,
    text: str,
    *,
    health_overall_ok: bool,
    context_for_risk: str,
    alert_on_degraded: bool = True,
    alert_on_high_risk: bool = True,
) -> CommandDecision:
    """Same parsing rules as legacy KirkAgent._parse (without SpockBriefing object)."""
    sys_status = "NOMINAL" if health_overall_ok else "DEGRADED"
    overall_risk = infer_overall_risk_from_text(context_for_risk)
    mkt_posture = _RISK_TO_POSTURE.get(overall_risk, "NORMAL")
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
        (alert_on_degraded and not health_overall_ok)
        or (alert_on_high_risk and overall_risk in ("HIGH", "CRITICAL"))
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


def save_decision(root: Path, decision: CommandDecision) -> None:
    path = root / "data" / "artifacts" / "crew_decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing: list = []
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.append(decision.to_json())
        existing = existing[-500:]
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception:
        pass


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
