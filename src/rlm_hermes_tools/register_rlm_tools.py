"""
Register RLM tools into Hermes' global ``tools.registry`` (toolset ``rlm``).

Import this module after ``run_agent`` / ``model_tools`` have loaded so
builtin tool discovery has already run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tools.registry import registry

from rlm.hermes_facts.health import gather_health_report
from rlm.hermes_facts.market_context import build_trade_and_regime_context
from rlm.roee.system_gate import SystemGate


def _root() -> Path:
    return Path(os.environ.get("RLM_ROOT", os.getcwd())).resolve()


def _rlm_get_health_report_json(args: dict | None = None, **kw) -> str:
    root = _root()
    return json.dumps(gather_health_report(root), ensure_ascii=False, default=str)


def _rlm_get_trade_and_regime_context_json(args: dict | None = None, **kw) -> str:
    root = _root()
    text = build_trade_and_regime_context(root)
    return json.dumps({"context": text}, ensure_ascii=False)


def _rlm_get_system_gate_state_json(args: dict | None = None, **kw) -> str:
    gate = SystemGate(_root())
    st = gate.load()
    return json.dumps(
        {
            "posture": st.posture,
            "status": st.status,
            "last_updated": st.last_updated,
            "trading_allowed": gate.is_trading_allowed(),
        },
        ensure_ascii=False,
    )


def _rlm_check_portfolio_limits_json(args: dict | None = None, **kw) -> str:
    """Surface sizing env caps for the model (deterministic facts)."""
    data = {
        "max_kelly_fraction": os.environ.get("MAX_KELLY_FRACTION", ""),
        "max_capital_fraction": os.environ.get("MAX_CAPITAL_FRACTION", ""),
        "note": "ROEE policy limits also live in YAML configs; these env vars are optional hints.",
    }
    return json.dumps({k: v for k, v in data.items() if v or k == "note"}, ensure_ascii=False)


def _check_rlm_root() -> bool:
    return _root().exists()


RLM_HEALTH_SCHEMA = {
    "name": "rlm_get_health_report",
    "description": (
        "Return JSON health facts for the RLM host: systemd units, disk, stale artefacts, "
        "recent journal errors, rlm doctor output, optional auto-restart log."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RLM_CONTEXT_SCHEMA = {
    "name": "rlm_get_trade_and_regime_context",
    "description": (
        "Return JSON with a single `context` string: active trade plans, latest pipeline run "
        "artefact, equity positions, walk-forward summaries (regime research snapshot)."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RLM_GATE_SCHEMA = {
    "name": "rlm_get_system_gate_state",
    "description": "Return JSON system gate posture/status from data/processed/gate_state.json.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RLM_LIMITS_SCHEMA = {
    "name": "rlm_check_portfolio_limits",
    "description": "Return JSON of configured portfolio / sizing hints from environment.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


registry.register(
    name="rlm_get_health_report",
    toolset="rlm",
    schema=RLM_HEALTH_SCHEMA,
    handler=lambda args, **kw: _rlm_get_health_report_json(args or {}, **kw),
    check_fn=_check_rlm_root,
    emoji="🔧",
)

registry.register(
    name="rlm_get_trade_and_regime_context",
    toolset="rlm",
    schema=RLM_CONTEXT_SCHEMA,
    handler=lambda args, **kw: _rlm_get_trade_and_regime_context_json(args or {}, **kw),
    check_fn=_check_rlm_root,
    emoji="📊",
)

registry.register(
    name="rlm_get_system_gate_state",
    toolset="rlm",
    schema=RLM_GATE_SCHEMA,
    handler=lambda args, **kw: _rlm_get_system_gate_state_json(args or {}, **kw),
    check_fn=_check_rlm_root,
    emoji="🚧",
)

registry.register(
    name="rlm_check_portfolio_limits",
    toolset="rlm",
    schema=RLM_LIMITS_SCHEMA,
    handler=lambda args, **kw: _rlm_check_portfolio_limits_json(args or {}, **kw),
    check_fn=lambda: True,
    emoji="⚖️",
)
