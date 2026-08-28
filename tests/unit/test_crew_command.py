"""Commander parse → system-gate kill-switch (STAND-DOWN must not lift on parse miss)."""

from __future__ import annotations

from pathlib import Path

from rlm.hermes_facts.crew_command import apply_crew_gate_update, parse_command_decision
from rlm.roee.system_gate import SystemGate

_TS = "2026-08-28T11:00:00Z"


def _parse(text: str, *, context: str = "Market State: rth", health_ok: bool = True):
    return parse_command_decision(
        _TS,
        text,
        health_overall_ok=health_ok,
        context_for_risk=context,
    )


def test_omitted_market_posture_defaults_display_normal_but_not_explicit() -> None:
    decision = _parse("COMMAND DECISION: HOLD\nRATIONALE: quiet tape.\n")
    assert decision.market_posture == "NORMAL"
    assert decision.posture_explicit is False
    assert decision.command == "HOLD"


def test_markdown_market_posture_is_explicit_stand_down() -> None:
    decision = _parse("**MARKET POSTURE:** STAND-DOWN\nCOMMAND DECISION: STAND-DOWN — vol spike\n")
    assert decision.market_posture == "STAND-DOWN"
    assert decision.posture_explicit is True


def test_command_stand_down_without_posture_line_is_explicit() -> None:
    decision = _parse("COMMAND DECISION: STAND-DOWN — operator halt\nRATIONALE: halt.\n")
    assert decision.market_posture == "STAND-DOWN"
    assert decision.posture_explicit is True


def test_parse_miss_does_not_lift_existing_stand_down(tmp_path: Path) -> None:
    gate = SystemGate(tmp_path)
    gate.update("STAND-DOWN", "NOMINAL", "2026-08-28T10:00:00Z")
    decision = _parse("COMMAND DECISION: HOLD\nRATIONALE: omitted posture line.\n")
    assert decision.posture_explicit is False
    apply_crew_gate_update(gate, decision)
    allowed, state = gate.check()
    assert allowed is False
    assert state.posture == "STAND-DOWN"


def test_parse_miss_does_not_lift_critical_status(tmp_path: Path) -> None:
    gate = SystemGate(tmp_path)
    gate.update("NORMAL", "CRITICAL", "2026-08-28T10:00:00Z")
    decision = _parse("MARKET POSTURE: NORMAL\nCOMMAND DECISION: HOLD\n", health_ok=True)
    assert decision.status_explicit is False
    apply_crew_gate_update(gate, decision)
    allowed, state = gate.check()
    assert allowed is False
    assert state.status == "CRITICAL"


def test_explicit_normal_can_lift_stand_down(tmp_path: Path) -> None:
    gate = SystemGate(tmp_path)
    gate.update("STAND-DOWN", "NOMINAL", "2026-08-28T10:00:00Z")
    decision = _parse("MARKET POSTURE: NORMAL\nSYSTEM STATUS: NOMINAL\nCOMMAND DECISION: GO\n")
    assert decision.posture_explicit is True
    apply_crew_gate_update(gate, decision)
    allowed, state = gate.check()
    assert allowed is True
    assert state.posture == "NORMAL"


def test_commander_overall_risk_critical_tightens_without_lifting(tmp_path: Path) -> None:
    gate = SystemGate(tmp_path)
    gate.update("NORMAL", "NOMINAL", "2026-08-28T10:00:00Z")
    decision = _parse("COMMAND DECISION: HOLD\nOVERALL RISK POSTURE: CRITICAL\n")
    assert decision.posture_explicit is False
    assert decision.market_posture == "STAND-DOWN"
    apply_crew_gate_update(gate, decision)
    allowed, state = gate.check()
    assert allowed is False
    assert state.posture == "STAND-DOWN"
