"""Parse Hermes commander LLM output and persist crew decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlm.roee.system_gate import SystemGate

_RISK_TO_POSTURE = {
    "LOW": "AGGRESSIVE",
    "MODERATE": "NORMAL",
    "HIGH": "DEFENSIVE",
    "CRITICAL": "STAND-DOWN",
    "UNKNOWN": "NORMAL",
}

_VALID_POSTURES = frozenset({"AGGRESSIVE", "NORMAL", "DEFENSIVE", "STAND-DOWN"})
_VALID_STATUSES = frozenset({"NOMINAL", "DEGRADED", "CRITICAL"})
_POSTURE_RANK = {"AGGRESSIVE": 0, "NORMAL": 1, "DEFENSIVE": 2, "STAND-DOWN": 3}
_STATUS_RANK = {"NOMINAL": 0, "DEGRADED": 1, "CRITICAL": 2}

_RISK_PATTERN = re.compile(
    r"OVERALL\s+RISK\s+POSTURE\s*:\s*(CRITICAL|HIGH|MODERATE|LOW)",
    re.IGNORECASE,
)
_LEADING_MARKDOWN = re.compile(r"^[\s>*#_+\-]*[\*_`]+")


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
    posture_explicit: bool = False
    status_explicit: bool = False

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


def _plain_decision_line(line: str) -> str:
    """Strip common LLM markdown wrapping so ``**MARKET POSTURE:**`` still parses."""
    s = line.strip()
    while True:
        nxt = _LEADING_MARKDOWN.sub("", s).strip()
        if nxt == s:
            break
        s = nxt
    return s


def _plain_field_value(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^[\*_`]+", "", s)
    s = re.sub(r"[\*_`]+$", "", s)
    return s.strip()


def parse_command_decision(
    ts: str,
    text: str,
    *,
    health_overall_ok: bool,
    context_for_risk: str,
    alert_on_degraded: bool = True,
    alert_on_high_risk: bool = True,
) -> CommandDecision:
    """Derive CommandDecision from commander-format plain text."""
    sys_status = "NOMINAL" if health_overall_ok else "DEGRADED"
    overall_risk = infer_overall_risk_from_text(text)
    if overall_risk == "UNKNOWN":
        overall_risk = infer_overall_risk_from_text(context_for_risk)
    mkt_posture = _RISK_TO_POSTURE.get(overall_risk, "NORMAL")
    command_str = "HOLD"
    rationale = "Defaulted due to LLM parse failure."
    orders: dict[str, str] = {}
    posture_explicit = False
    status_explicit = False

    for line in text.splitlines():
        stripped = _plain_decision_line(line)
        if stripped.startswith("SYSTEM STATUS:"):
            val = _plain_field_value(stripped.split(":", 1)[1]).upper()
            if val in _VALID_STATUSES:
                sys_status = val
                status_explicit = True
        elif stripped.startswith("MARKET POSTURE:"):
            val = _plain_field_value(stripped.split(":", 1)[1]).upper()
            if val in _VALID_POSTURES:
                mkt_posture = val
                posture_explicit = True
        elif stripped.startswith("COMMAND DECISION:"):
            command_str = _plain_field_value(stripped.split(":", 1)[1])
        elif stripped.startswith("RATIONALE:"):
            rationale = _plain_field_value(stripped.split(":", 1)[1])
        elif stripped.startswith("- ") and ":" in stripped:
            key, _, val = stripped[2:].partition(":")
            orders[key.strip()] = val.strip()

    if not posture_explicit and command_str.strip().upper().startswith("STAND-DOWN"):
        mkt_posture = "STAND-DOWN"
        posture_explicit = True

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
        posture_explicit=posture_explicit,
        status_explicit=status_explicit,
    )


def apply_crew_gate_update(gate: SystemGate, decision: CommandDecision) -> None:
    """Persist commander posture/status without lifting a halt on a parse miss.

    ``regime-locus-crew`` overwrites ``gate_state.json`` every briefing. If the LLM
    omits ``MARKET POSTURE:`` (or wraps it in markdown), the parser used to default
    UNKNOWN→NORMAL and resume options trading while STAND-DOWN was still in force.
    """
    existing = gate.load()
    posture = decision.market_posture
    status = decision.system_status
    if posture not in _VALID_POSTURES:
        posture = existing.posture
    if status not in _VALID_STATUSES:
        status = existing.status

    if not decision.posture_explicit:
        if _POSTURE_RANK.get(posture, 1) < _POSTURE_RANK.get(existing.posture, 1):
            posture = existing.posture
    if not decision.status_explicit:
        if _STATUS_RANK.get(status, 0) < _STATUS_RANK.get(existing.status, 0):
            status = existing.status

    gate.update(posture=posture, status=status, timestamp=decision.timestamp)


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
