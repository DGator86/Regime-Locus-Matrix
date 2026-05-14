"""
Global System Gate — manages system posture and trading permissions.

Persists ``data/processed/gate_state.json`` with keys:
``posture``, ``status``, ``last_updated``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateState:
    posture: str = "NORMAL"  # AGGRESSIVE | NORMAL | DEFENSIVE | STAND-DOWN
    status: str = "NOMINAL"  # NOMINAL | DEGRADED | CRITICAL
    last_updated: str = ""


class SystemGate:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "data" / "processed" / "gate_state.json"

    def load(self) -> GateState:
        if not self.path.is_file():
            return GateState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return GateState(**data)
        except Exception:
            return GateState()

    def update(self, posture: str, status: str, timestamp: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = GateState(posture=posture, status=status, last_updated=timestamp)
        self.path.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")

    def is_trading_allowed(self) -> bool:
        state = self.load()
        return state.status != "CRITICAL" and state.posture != "STAND-DOWN"

    def check(self) -> tuple[bool, GateState]:
        """Return (trading_allowed, state) from a single load — avoids redundant reads."""
        state = self.load()
        allowed = state.status != "CRITICAL" and state.posture != "STAND-DOWN"
        return allowed, state
