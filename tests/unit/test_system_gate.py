"""System gate path and trading permission (moved from rlm.agents.gate)."""

from __future__ import annotations

import json
from pathlib import Path

from rlm.roee.system_gate import GateState, SystemGate


def test_system_gate_path_and_schema(tmp_path: Path) -> None:
    gate = SystemGate(tmp_path)
    assert gate.path == tmp_path / "data" / "processed" / "gate_state.json"
    gate.update("NORMAL", "NOMINAL", "2020-01-01T00:00:00Z")
    data = json.loads(gate.path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"posture", "status", "last_updated"}
    assert gate.is_trading_allowed() is True
    gate.update("STAND-DOWN", "NOMINAL", "2020-01-02T00:00:00Z")
    assert gate.is_trading_allowed() is False
    gate.update("NORMAL", "CRITICAL", "2020-01-03T00:00:00Z")
    assert gate.is_trading_allowed() is False


def test_gate_state_defaults() -> None:
    st = GateState()
    assert st.posture == "NORMAL"
    assert st.status == "NOMINAL"
