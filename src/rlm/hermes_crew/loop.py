"""Hermes AIAgent loop — three-agent crew: Scotty (data_monitor) → Spock (research_analyst) → Kirk (commander)."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from rlm.hermes_facts.health import gather_health_report
from rlm.hermes_facts.kirk_command import (
    CommandDecision,
    parse_command_decision,
    save_decision,
    utc_timestamp,
)
from rlm.hermes_facts.market_context import build_trade_and_regime_context
from rlm.roee.system_gate import SystemGate
from rlm.utils.telegram_crew_notify import resolve_telegram_chat_id, telegram_crew_send

_KIRK_SYSTEM = """\
You are Captain Kirk, the commanding officer of this trading system.
You have received reports from your Chief Engineer (Scotty) and Science Officer (Spock).
Your role: make the final command decision and communicate it clearly to the crew.

You may call the rlm_* tools to refresh live facts from the trading host before deciding.

SYSTEM HOURS:
- Market State: [rth / pre_market / after_hours / weekend]
- If the state is 'after_hours' or 'weekend', the starship is in power-save mode.
- "Everything being dark" (services offline) is INTENDED and NORMAL.
- Maintain a STAND-DOWN posture and HOLD command.
- Do not alert the operator for expected after-hours service closures.

Response format (plain text, no markdown):
SYSTEM STATUS: [NOMINAL / DEGRADED / CRITICAL]
MARKET POSTURE: [AGGRESSIVE / NORMAL / DEFENSIVE / STAND-DOWN]
COMMAND DECISION: <one decisive sentence — GO / HOLD / STAND-DOWN / ALERT OPERATOR>
RATIONALE: <2-3 sentences max, referencing Scotty and Spock's key findings>
CREW ORDERS:
  - Scotty: <one action item or "maintain current status">
  - Spock: <one action item or "continue monitoring">
  - Helm: <one directive for the trading engine>
"""

_SCOTTY_FALLBACK = """\
You are Scotty, Chief Engineer of this trading system. You are practical and direct.
When the market is closed, do not panic about powered-down batch services.
Summarise what is broken, what is fine, and what to do next in 3-10 short bullets (plain text, no markdown headers).
"""

_SPOCK_FALLBACK = """\
You are Spock: logical, probability-focused, no emotional language.
Analyse active trade plans and regime signals. Number each active plan:
  1. SYMBOL | STRATEGY | REGIME | ACTION: [GO / HOLD / ABORT] | RATIONALE: <one sentence>
End with: OVERALL RISK POSTURE: [LOW / MODERATE / HIGH / CRITICAL]
"""


@dataclass
class HermesCrewConfig:
    health_interval: int = int(os.environ.get("CREW_HEALTH_INTERVAL", "120"))
    analysis_interval: int = int(os.environ.get("CREW_ANALYSIS_INTERVAL", "300"))
    briefing_interval: int = int(os.environ.get("CREW_BRIEFING_INTERVAL", "600"))
    telegram_token: str = (
        os.environ.get("RLM_HERMES_TELEGRAM_BOT_TOKEN")
        or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = (
        os.environ.get("RLM_HERMES_TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    )
    silent_health_ok: bool = True


def _load_skill_text(root: Path, skill_name: str, fallback: str) -> str:
    p = root / "hermes_skills" / skill_name / "SKILL.md"
    if not p.is_file():
        return fallback
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return raw.strip() or fallback


def _load_commander_skill_text(root: Path) -> str:
    return _load_skill_text(root, "commander", _KIRK_SYSTEM)


def _load_scotty_skill_text(root: Path) -> str:
    return _load_skill_text(root, "data_monitor", _SCOTTY_FALLBACK)


def _load_spock_skill_text(root: Path) -> str:
    return _load_skill_text(root, "research_analyst", _SPOCK_FALLBACK)


def _hermes_updates_system_gate() -> bool:
    """When false, Kirk/Hermes still briefs but does not overwrite gate_state.json."""
    v = (os.environ.get("RLM_HERMES_UPDATE_GATE") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _ensure_hermes(root: Path) -> Tuple[Any, Any]:
    os.environ.setdefault("RLM_ROOT", str(root))
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    try:
        import run_agent  # noqa: WPS433 — third-party entry
        import rlm_hermes_tools.register_rlm_tools  # noqa: F401, WPS433 — registers tools
    except ImportError as e:
        raise RuntimeError(
            "Hermes agent is not installed. Install with: pip install -e \".[hermes]\""
        ) from e
    return run_agent.AIAgent, run_agent


def _env_first(*keys: str) -> str:
    for key in keys:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def _resolve_hermes_backends() -> list[tuple[str, str, str]]:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key and not os.environ.get("RLM_HERMES_BASE_URL", "").strip():
        _dflt_base = "https://api.groq.com/openai/v1"
        _dflt_key = groq_key
        _dflt_model = "llama-3.1-8b-instant"
    else:
        _dflt_base = "http://127.0.0.1:11434/v1"
        _dflt_key = "ollama"
        _dflt_model = "llama3.2"

    primary_base = _env_first("RLM_HERMES_BASE_URL") or _dflt_base
    primary_key = _env_first("RLM_HERMES_API_KEY") or _dflt_key
    primary_model = _env_first("RLM_HERMES_MODEL", "LLM_MODEL") or _dflt_model

    fallback_model = _env_first("RLM_HERMES_FALLBACK_MODEL")
    fallback_base = _env_first("RLM_HERMES_FALLBACK_BASE_URL")
    fallback_key = _env_first("RLM_HERMES_FALLBACK_API_KEY")
    if fallback_model and not fallback_base:
        fallback_base = "https://openrouter.ai/api/v1"
    if fallback_base and not fallback_key:
        fallback_key = _env_first("OPENROUTER_API_KEY")

    backends: list[tuple[str, str, str]] = [(primary_base, primary_key, primary_model)]
    if fallback_base and fallback_key and fallback_model:
        backends.append((fallback_base, fallback_key, fallback_model))
    return backends


def _make_agent_with_skill(
    root: Path,
    skill_prompt: str,
    toolsets: list[str],
    backend: tuple[str, str, str],
):
    AIAgent, _ = _ensure_hermes(root)
    base_url, api_key, model = backend
    skip_memory = os.environ.get("RLM_HERMES_SKIP_MEMORY", "").strip().lower() in ("1", "true", "yes")
    max_it = int(os.environ.get("RLM_HERMES_MAX_ITERATIONS", "20"))
    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        quiet_mode=True,
        enabled_toolsets=toolsets,
        ephemeral_system_prompt=skill_prompt,
        skip_memory=skip_memory,
        max_iterations=max_it,
        skip_context_files=True,
    )


def _chat_with_failover(root: Path, skill_prompt: str, user_prompt: str, toolsets: list[str]) -> str:
    backends = _resolve_hermes_backends()
    last_error: Exception | None = None
    for idx, backend in enumerate(backends, start=1):
        base_url, _, model = backend
        try:
            if idx > 1:
                print(
                    f"[Hermes crew] retrying with fallback backend #{idx}: {base_url} model={model}",
                    flush=True,
                )
            agent = _make_agent_with_skill(root, skill_prompt, toolsets, backend)
            out = agent.chat(user_prompt)
            if out:
                return out
            raise RuntimeError("Hermes returned empty response")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[Hermes crew] backend #{idx} failed ({base_url} model={model}): {exc}",
                flush=True,
            )
            continue
    if last_error is None:
        raise RuntimeError("No Hermes backends configured")
    raise RuntimeError(f"All Hermes backends failed: {last_error}")


def _make_agent(root: Path):
    """Commander agent (Kirk). Kept for backward compatibility."""
    return _make_agent_with_skill(
        root,
        _load_commander_skill_text(root),
        ["rlm"],
    )


def _run_scotty_agent(root: Path, health_facts_json: str) -> str:
    """Run the Scotty (data_monitor) Hermes agent; returns its plain-text engineering report."""
    return _chat_with_failover(
        root,
        _load_scotty_skill_text(root),
        f"Here are the raw system health facts (JSON):\n\n{health_facts_json}\n\n"
        "Call rlm_get_health_report or rlm_get_system_gate_state if you need fresher data. "
        "Produce your engineering report now.",
        ["rlm"],
    )


def _run_spock_agent(root: Path, market_context: str) -> str:
    """Run the Spock (research_analyst) Hermes agent; returns its plain-text analysis."""
    return _chat_with_failover(
        root,
        _load_spock_skill_text(root),
        f"Here is the current market context:\n\n{market_context}\n\n"
        "Call rlm_get_trade_and_regime_context, rlm_get_system_gate_state, or "
        "rlm_check_portfolio_limits if you need fresher data. "
        "Produce your analysis now.",
        ["rlm"],
    )


def _run_full_briefing(
    root: Path,
    health_payload: dict,
    market_context: str,
) -> tuple[str, str, str]:
    """Run all three Hermes agents in sequence: Scotty → Spock → Kirk.

    Returns (scotty_report, spock_report, kirk_llm_text).
    """
    import json

    health_json = json.dumps(health_payload, default=str)

    print("[Hermes crew] Running Scotty (data_monitor) agent...", flush=True)
    scotty_report = _run_scotty_agent(root, health_json)
    print(f"[Hermes crew] Scotty done ({len(scotty_report)} chars)", flush=True)

    print("[Hermes crew] Running Spock (research_analyst) agent...", flush=True)
    spock_report = _run_spock_agent(root, market_context)
    print(f"[Hermes crew] Spock done ({len(spock_report)} chars)", flush=True)

    print("[Hermes crew] Running Kirk (commander) agent...", flush=True)
    kirk_user = (
        "Here are your crew reports.\n\n"
        f"=== Scotty's Engineering Report ===\n{scotty_report}\n\n"
        f"=== Spock's Market Analysis ===\n{spock_report}\n\n"
        "You may call rlm_get_health_report, rlm_get_trade_and_regime_context, "
        "rlm_get_system_gate_state, or rlm_check_portfolio_limits if you need fresher data.\n\n"
        "Issue your command decision in the required format."
    )
    kirk_text = _chat_with_failover(root, _load_commander_skill_text(root), kirk_user, ["rlm"])
    print(f"[Hermes crew] Kirk done ({len(kirk_text)} chars)", flush=True)

    return scotty_report, spock_report, kirk_text


def run_crew_once(root: Path, cfg: Optional[HermesCrewConfig] = None) -> CommandDecision:
    cfg = cfg or HermesCrewConfig()
    os.environ["RLM_ROOT"] = str(root.resolve())
    health = gather_health_report(root)
    ctx = build_trade_and_regime_context(root)
    health_ok = bool(health.get("overall_ok", True))

    _, _, llm_text = _run_full_briefing(root, health, ctx)

    ts = utc_timestamp()
    decision = parse_command_decision(
        ts,
        llm_text,
        health_overall_ok=health_ok,
        context_for_risk=ctx,
    )
    save_decision(root, decision)
    if _hermes_updates_system_gate():
        gate = SystemGate(root)
        gate.update(
            posture=decision.market_posture,
            status=decision.system_status,
            timestamp=decision.timestamp,
        )
    if decision.system_status == "CRITICAL" and "ALERT OPERATOR" in decision.command.upper():
        cid = (cfg.telegram_chat_id or "").strip() or resolve_telegram_chat_id(root)
        telegram_crew_send(
            decision.to_telegram_message(),
            cfg.telegram_token,
            cid,
            silent=False,
        )
    return decision


def run_crew_forever(root: Path, cfg: Optional[HermesCrewConfig] = None) -> None:
    cfg = cfg or HermesCrewConfig()
    root = root.resolve()
    os.environ["RLM_ROOT"] = str(root)
    print(
        f"[Hermes crew] root={root} health={cfg.health_interval}s "
        f"analysis={cfg.analysis_interval}s briefing={cfg.briefing_interval}s",
        flush=True,
    )
    gate = SystemGate(root)
    last_health = 0.0
    last_analysis = 0.0
    last_briefing = 0.0
    last_health_payload: dict = {}
    last_context = ""
    eod_sent = False

    while True:
        now = time.monotonic()
        try:
            if now - last_health >= cfg.health_interval:
                last_health = now
                last_health_payload = gather_health_report(root)
                print(last_health_payload.get("report_text", ""), flush=True)

            if now - last_analysis >= cfg.analysis_interval:
                last_analysis = now
                last_context = build_trade_and_regime_context(root)

            if now - last_briefing >= cfg.briefing_interval:
                last_briefing = now
                if not last_health_payload:
                    last_health_payload = gather_health_report(root)
                if not last_context.strip():
                    last_context = build_trade_and_regime_context(root)
                health_ok = bool(last_health_payload.get("overall_ok", True))

                _, _, llm_text = _run_full_briefing(root, last_health_payload, last_context)
                ts = utc_timestamp()
                decision = parse_command_decision(
                    ts,
                    llm_text,
                    health_overall_ok=health_ok,
                    context_for_risk=last_context,
                )
                save_decision(root, decision)
                if _hermes_updates_system_gate():
                    gate.update(
                        posture=decision.market_posture,
                        status=decision.system_status,
                        timestamp=decision.timestamp,
                    )
                print(
                    f"[Hermes crew] Command: {decision.command} "
                    f"(Posture: {decision.market_posture})",
                    flush=True,
                )
                if decision.system_status == "CRITICAL" and "ALERT OPERATOR" in decision.command.upper():
                    cid = (cfg.telegram_chat_id or "").strip() or resolve_telegram_chat_id(root)
                    telegram_crew_send(
                        decision.to_telegram_message(),
                        cfg.telegram_token,
                        cid,
                        silent=False,
                    )
        except Exception as exc:
            print(f"[Hermes crew ERROR] {exc}", flush=True)

        now_utc = datetime.now(timezone.utc)
        if now_utc.hour == 20 and 15 <= now_utc.minute < 30:
            if not eod_sent:
                try:
                    from rlm.notify.pnl_report import calculate_daily_pnl

                    report_text = calculate_daily_pnl(root)
                    cid = (cfg.telegram_chat_id or "").strip() or resolve_telegram_chat_id(root)
                    telegram_crew_send(
                        report_text,
                        cfg.telegram_token,
                        cid,
                        silent=False,
                    )
                    eod_sent = True
                except Exception as exc:
                    print(f"[EOD ERROR] {exc}", flush=True)
        elif now_utc.hour != 20:
            eod_sent = False

        time.sleep(1.0)
