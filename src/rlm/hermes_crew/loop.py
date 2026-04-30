"""Hermes AIAgent loop replacing StarfleetCrew (Kirk briefing + gate + Telegram)."""

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


@dataclass
class HermesCrewConfig:
    health_interval: int = int(os.environ.get("CREW_HEALTH_INTERVAL", "120"))
    analysis_interval: int = int(os.environ.get("CREW_ANALYSIS_INTERVAL", "300"))
    briefing_interval: int = int(os.environ.get("CREW_BRIEFING_INTERVAL", "600"))
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    silent_health_ok: bool = True


def _load_commander_skill_text(root: Path) -> str:
    p = root / "hermes_skills" / "commander" / "SKILL.md"
    if not p.is_file():
        return _KIRK_SYSTEM
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return raw.strip() or _KIRK_SYSTEM


def _ensure_hermes(root: Path) -> Tuple[Any, Any]:
    os.environ.setdefault("RLM_ROOT", str(root))
    sys.path.append(str(root))
    try:
        import run_agent  # noqa: WPS433 — third-party entry
        import rlm_hermes_tools.register_rlm_tools  # noqa: F401, WPS433 — registers tools
    except ImportError as e:
        raise RuntimeError(
            "Hermes agent is not installed. Install with: pip install -e \".[hermes]\""
        ) from e
    return run_agent.AIAgent, run_agent


def _make_agent(root: Path):
    AIAgent, _ = _ensure_hermes(root)
    base_url = os.environ.get("RLM_HERMES_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("RLM_HERMES_API_KEY", "ollama")
    model = os.environ.get("RLM_HERMES_MODEL", os.environ.get("LLM_MODEL", "llama3.2"))
    skip_memory = os.environ.get("RLM_HERMES_SKIP_MEMORY", "").strip().lower() in ("1", "true", "yes")
    max_it = int(os.environ.get("RLM_HERMES_MAX_ITERATIONS", "20"))
    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=model,
        quiet_mode=True,
        enabled_toolsets=["rlm"],
        ephemeral_system_prompt=_load_commander_skill_text(root),
        skip_memory=skip_memory,
        max_iterations=max_it,
        skip_context_files=True,
    )


def run_crew_once(root: Path, cfg: Optional[HermesCrewConfig] = None) -> CommandDecision:
    cfg = cfg or HermesCrewConfig()
    os.environ["RLM_ROOT"] = str(root.resolve())
    health = gather_health_report(root)
    ctx = build_trade_and_regime_context(root)
    health_ok = bool(health.get("overall_ok", True))
    health_txt = str(health.get("report_text", ""))
    agent = _make_agent(root)
    user = (
        "Here are your crew reports (cached at invocation time).\n\n"
        f"=== Scotty's Engineering Report ===\n{health_txt}\n\n"
        f"=== Spock's Market Context ===\n{ctx}\n\n"
        "You may call rlm_get_health_report, rlm_get_trade_and_regime_context, "
        "rlm_get_system_gate_state, or rlm_check_portfolio_limits if you need fresher data.\n\n"
        "Issue your command decision in the required format."
    )
    llm_text = agent.chat(user)
    ts = utc_timestamp()
    decision = parse_command_decision(
        ts,
        llm_text,
        health_overall_ok=health_ok,
        context_for_risk=ctx,
    )
    save_decision(root, decision)
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
                health_txt = str(last_health_payload.get("report_text") or "")
                if not health_txt.strip():
                    last_health_payload = gather_health_report(root)
                    health_txt = str(last_health_payload.get("report_text", ""))
                if not last_context.strip():
                    last_context = build_trade_and_regime_context(root)
                health_ok = bool(last_health_payload.get("overall_ok", True))

                agent = _make_agent(root)
                user = (
                    "Here are your crew reports:\n\n"
                    f"=== Scotty's Engineering Report ===\n{health_txt}\n\n"
                    f"=== Spock's Market Context ===\n{last_context}\n\n"
                    "You may call rlm_* tools for fresher data.\n\n"
                    "Issue your command decision in the required format."
                )
                llm_text = agent.chat(user)
                ts = utc_timestamp()
                decision = parse_command_decision(
                    ts,
                    llm_text,
                    health_overall_ok=health_ok,
                    context_for_risk=last_context,
                )
                save_decision(root, decision)
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
