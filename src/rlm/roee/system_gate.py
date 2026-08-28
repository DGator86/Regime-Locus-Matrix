"""
Global System Gate — manages system posture and trading permissions.

Persists ``data/processed/gate_state.json`` with keys:
``posture``, ``status``, ``last_updated``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

_VALID_POSTURES = frozenset({"AGGRESSIVE", "NORMAL", "DEFENSIVE", "STAND-DOWN"})
_VALID_STATUSES = frozenset({"NOMINAL", "DEGRADED", "CRITICAL"})


@dataclass
class GateState:
    posture: str = "NORMAL"  # AGGRESSIVE | NORMAL | DEFENSIVE | STAND-DOWN
    status: str = "NOMINAL"  # NOMINAL | DEGRADED | CRITICAL
    last_updated: str = ""


def _halted_state(last_updated: str = "") -> GateState:
    """Fail-closed sentinel when on-disk gate state cannot be trusted."""
    return GateState(posture="STAND-DOWN", status="CRITICAL", last_updated=last_updated)


class SystemGate:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "data" / "processed" / "gate_state.json"

    def load(self) -> GateState:
        if not self.path.is_file():
            return GateState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return _halted_state()
            posture = str(data.get("posture") or "").strip().upper()
            status = str(data.get("status") or "").strip().upper()
            last_updated = str(data.get("last_updated") or "")
            if posture not in _VALID_POSTURES or status not in _VALID_STATUSES:
                return _halted_state(last_updated)
            return GateState(posture=posture, status=status, last_updated=last_updated)
        except Exception:
            return _halted_state()

    def update(self, posture: str, status: str, timestamp: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = GateState(posture=posture, status=status, last_updated=timestamp)
        payload = json.dumps(state.__dict__, indent=2)
        tmp = self.path.parent / f"{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def is_trading_allowed(self) -> bool:
        state = self.load()
        return state.status != "CRITICAL" and state.posture != "STAND-DOWN"

    def check(self) -> tuple[bool, GateState]:
        """Return (trading_allowed, state) from a single load — avoids redundant reads."""
        state = self.load()
        allowed = state.status != "CRITICAL" and state.posture != "STAND-DOWN"
        return allowed, state
