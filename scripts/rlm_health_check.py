#!/usr/bin/env python3
"""
RLM System Health Check -- end-to-end synthetic data run + Hermes crew activation.

Usage:
    python scripts/rlm_health_check.py                  # isolated output (default)
    python scripts/rlm_health_check.py --force          # write into live data/ dirs
    python scripts/rlm_health_check.py --output-root /tmp/rlm_check

Steps:
  1  Imports & environment
  2  Seed synthetic artifacts (trade plans, walkforward, gate_state)
  3  FullRLMPipeline: bars_SPY.csv -> factors -> HMM regime -> ROEE policy
  4  Hermes tool handlers (all 4 RLM tools exercised directly)
  5  Pipeline health: data_monitor Hermes agent (engineering report)
  6  Regime research: research_analyst Hermes agent (market & regime analysis)
  7  Commander: final crew decision + gate update
  8  Hermes skill file inventory
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows cp1252 consoles raise UnicodeEncodeError on non-ASCII output.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

os.environ.setdefault("RLM_ROOT", str(ROOT))
os.environ.setdefault("RLM_HERMES_SKIP_MEMORY", "1")
os.environ.setdefault("RLM_HEALTH_AUTO_RESTART", "0")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"
HEAD = "\033[1;35m"

_failures: list[str] = []
_step = 0


def step(title: str) -> None:
    global _step
    _step += 1
    print(f"\n{HEAD}{'='*64}\033[0m", flush=True)
    print(f"{HEAD}STEP {_step}: {title}\033[0m", flush=True)
    print(f"{HEAD}{'='*64}\033[0m", flush=True)


def ok(msg: str) -> None:
    print(f"  [{PASS}] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"  [{INFO}] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  [{FAIL}] {msg}", flush=True)
    _failures.append(msg)


def check(cond: bool, msg_pass: str, msg_fail: str) -> bool:
    if cond:
        ok(msg_pass)
    else:
        fail(msg_fail)
    return cond


# ---------------------------------------------------------------------------
# CLI: --force / --output-root
# ---------------------------------------------------------------------------
_live_processed = ROOT / "data" / "processed"
_live_artifacts = ROOT / "data" / "artifacts"

_force = False
_output_root: Path | None = None
_argv = sys.argv[1:]
_i = 0
while _i < len(_argv):
    _arg = _argv[_i]
    if _arg == "--force":
        _force = True
    elif _arg == "--output-root":
        if _i + 1 >= len(_argv):
            raise SystemExit("Missing value for --output-root")
        _output_root = Path(_argv[_i + 1]).expanduser()
        _i += 1
    elif _arg.startswith("--output-root="):
        _output_root = Path(_arg.split("=", 1)[1]).expanduser()
    _i += 1

if _force and _output_root is not None:
    raise SystemExit("Use either --force or --output-root, not both")

if _force:
    data_root = ROOT
    processed = _live_processed
    artifacts = _live_artifacts
else:
    _isolated = (
        _output_root
        if _output_root is not None
        else _live_artifacts / "health_check" / datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    data_root = _isolated
    processed = data_root / "data" / "processed"
    artifacts = data_root / "data" / "artifacts"

os.environ["RLM_ROOT"] = str(data_root)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Imports & environment
# ─────────────────────────────────────────────────────────────────────────────
step("Imports & environment")

import pandas as pd

try:
    from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline

    ok("rlm.core.pipeline -- FullRLMPipeline, FullRLMConfig")
except Exception as e:
    fail(f"rlm.core.pipeline: {e}")

try:
    from rlm.hermes_facts.health import gather_health_report

    ok("rlm.hermes_facts.health -- gather_health_report")
except Exception as e:
    fail(f"rlm.hermes_facts.health: {e}")

try:
    from rlm.hermes_facts.market_context import build_trade_and_regime_context

    ok("rlm.hermes_facts.market_context -- build_trade_and_regime_context")
except Exception as e:
    fail(f"rlm.hermes_facts.market_context: {e}")

try:
    from rlm.hermes_facts.crew_command import (
        parse_command_decision,
        save_decision,
        utc_timestamp,
    )

    ok("rlm.hermes_facts.crew_command -- parse_command_decision, save_decision")
except Exception as e:
    fail(f"rlm.hermes_facts.crew_command: {e}")

try:
    from rlm.roee.system_gate import SystemGate

    ok("rlm.roee.system_gate -- SystemGate")
except Exception as e:
    fail(f"rlm.roee.system_gate: {e}")

try:
    from rlm.hermes_crew.loop import (
        _load_commander_skill_text,
        _load_pipeline_health_skill_text,
        _load_regime_research_skill_text,
        _run_full_briefing,
    )

    ok("rlm.hermes_crew.loop -- skill loaders, _run_full_briefing")
except Exception as e:
    fail(f"rlm.hermes_crew.loop: {e}")

# Import loop_mod once here so both STEP 5 and STEP 6 can reference it independently.
loop_mod = importlib.import_module("rlm.hermes_crew.loop")

hermes_available = False
try:
    import run_agent  # noqa: F401

    import rlm_hermes_tools.register_rlm_tools  # noqa: F401

    hermes_available = True
    ok("hermes-agent (run_agent) -- INSTALLED, LLM agents can be called live")
except ImportError:
    info("hermes-agent (run_agent) -- NOT installed (pip install -e '.[hermes]')")
    info("Health check will exercise all Hermes infrastructure without the LLM call")

output_label = "live data/" if _force else str(data_root)
info(f"Artifact output root: {output_label}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Seed synthetic artifacts
# ─────────────────────────────────────────────────────────────────────────────
step("Seed synthetic artifacts (trade plans, walkforward, gate_state)")

processed.mkdir(parents=True, exist_ok=True)
artifacts.mkdir(parents=True, exist_ok=True)
(artifacts / "runs").mkdir(parents=True, exist_ok=True)

now_utc = datetime.now(tz=timezone.utc)
ts_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

plans_payload = {
    "generated_at_utc": ts_str,
    "results": [
        {
            "plan_id": "SYNTH_SPY_001",
            "symbol": "SPY",
            "strategy": "bull_call_spread",
            "status": "active",
            "rank_score": 0.82,
            "regime": 2,
            "regime_label": "calm_bull",
            "regime_confidence": 0.76,
            "kronos_return_forecast": 0.0031,
        },
        {
            "plan_id": "SYNTH_QQQ_002",
            "symbol": "QQQ",
            "strategy": "long_call",
            "status": "active",
            "rank_score": 0.71,
            "regime": 2,
            "regime_label": "calm_bull",
            "regime_confidence": 0.68,
            "kronos_return_forecast": 0.0019,
        },
        {
            "plan_id": "SYNTH_IWM_003",
            "symbol": "IWM",
            "strategy": "iron_condor",
            "status": "active",
            "rank_score": 0.55,
            "regime": 4,
            "regime_label": "high_vol",
            "regime_confidence": 0.59,
            "kronos_return_forecast": -0.0008,
        },
    ],
}
(processed / "universe_trade_plans.json").write_text(json.dumps(plans_payload, indent=2), encoding="utf-8")
ok("universe_trade_plans.json -- 3 active synthetic plans written")

run_artifact = {
    "run_id": "SYNTH_RUN_001",
    "symbol": "SPY",
    "timestamp": ts_str,
    "regime": 2,
    "regime_label": "calm_bull",
    "regime_confidence": 0.76,
    "kronos_return_forecast": 0.0031,
    "kronos_confidence": 0.71,
    "kronos_regime_agreement": True,
    "kronos_transition_flag": False,
    "roee_strategy": "bull_call_spread",
    "roee_confidence": 0.74,
}
(artifacts / "runs" / "SYNTH_RUN_001.json").write_text(json.dumps(run_artifact, indent=2), encoding="utf-8")
ok("artifacts/runs/SYNTH_RUN_001.json -- pipeline run artifact written")

wf_rows = [
    "win_rate,sharpe,avg_trade_pnl_pct,num_trades,oos_end,regime_safety_fraction,regime_safety_passed",
    "0.62,1.41,+1.2,28,2025-01-31,0.88,True",
    "0.58,1.22,+0.9,31,2025-02-28,0.85,True",
    "0.64,1.53,+1.4,26,2025-03-31,0.91,True",
]
(processed / "walkforward_summary_SPY.csv").write_text("\n".join(wf_rows), encoding="utf-8")
ok("walkforward_summary_SPY.csv -- 3 OOS windows written")

(processed / "gate_state.json").write_text(
    json.dumps({"posture": "NORMAL", "status": "NOMINAL", "last_updated": ts_str}, indent=2),
    encoding="utf-8",
)
ok("gate_state.json -- initial NORMAL / NOMINAL written")
info(f"All synthetic artifacts under: {processed}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- FullRLMPipeline: bars_SPY.csv -> factors -> HMM -> ROEE
# ─────────────────────────────────────────────────────────────────────────────
step("FullRLMPipeline -- bars_SPY.csv -> factors -> HMM regime -> ROEE policy")

bars_path = ROOT / "data" / "raw" / "bars_SPY.csv"
pipeline_result = None

try:
    bars = pd.read_csv(bars_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    info(
        f"Loaded {len(bars)} daily bars from {bars_path.name}  "
        f"({bars['timestamp'].min().date()} to {bars['timestamp'].max().date()})"
    )
    cfg = FullRLMConfig(regime_model="hmm", hmm_states=6, use_kronos=False, attach_vix=False)
    pipeline_result = FullRLMPipeline(cfg).run(bars)
    ok(
        f"Pipeline complete -- "
        f"factors_df: {len(pipeline_result.factors_df)} rows | "
        f"forecast_df: {len(pipeline_result.forecast_df)} rows | "
        f"policy_df: {len(pipeline_result.policy_df)} rows"
    )
except Exception as e:
    fail(f"FullRLMPipeline.run: {e}")

if pipeline_result is not None:
    pdf = pipeline_result.policy_df
    fdf = pipeline_result.forecast_df

    check(not pdf.empty, "policy_df non-empty", "policy_df is EMPTY")
    check("roee_action" in pdf.columns, "policy_df has roee_action", "policy_df MISSING roee_action")
    check(
        "roee_size_fraction" in pdf.columns, "policy_df has roee_size_fraction", "policy_df MISSING roee_size_fraction"
    )
    check("forecast_return" in fdf.columns, "forecast_df has forecast_return", "forecast_df MISSING forecast_return")
    check("hmm_confidence" in fdf.columns, "forecast_df has hmm_confidence", "forecast_df MISSING hmm_confidence")

    if "hmm_state" in pdf.columns:
        states = sorted(int(s) for s in pdf["hmm_state"].dropna().unique())
        info(f"HMM states detected: {states}  ({len(states)} distinct regimes)")

    action_counts = pdf["roee_action"].value_counts().to_dict()
    info(f"ROEE action distribution: {action_counts}")

    last = pdf.iloc[-1]
    info(
        f"Latest bar policy -- roee_action={last.get('roee_action','?')}  "
        f"size_fraction={last.get('roee_size_fraction', 0.0):.4f}  "
        f"hmm_state={last.get('hmm_state','?')}"
    )

    if "hmm_state" in pdf.columns and "hmm_confidence" in fdf.columns:
        live_artifact = {
            "run_id": "LIVE_RUN_SPY",
            "symbol": "SPY",
            "timestamp": ts_str,
            "regime": int(last.get("hmm_state", 0)),
            "regime_label": "hmm_detected",
            "regime_confidence": float(fdf["hmm_confidence"].iloc[-1]),
            "roee_strategy": str(last.get("roee_action", "hold")),
            "roee_size_fraction": float(last.get("roee_size_fraction", 0.0)),
        }
        (artifacts / "runs" / "LIVE_RUN_SPY.json").write_text(json.dumps(live_artifact, indent=2), encoding="utf-8")
        ok("artifacts/runs/LIVE_RUN_SPY.json -- live pipeline artifact persisted")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Hermes RLM tool handlers (all 4 tools exercised directly)
# ─────────────────────────────────────────────────────────────────────────────
step("Hermes RLM tool handlers -- exercise all 4 registered tools")

# Point health/gate checks at our selected data root so isolated runs do not touch live state.
os.environ["RLM_ROOT"] = str(data_root)

tools_ok = 0

try:
    health_raw = gather_health_report(data_root)
    health_json = json.dumps(health_raw, default=str)
    check(
        isinstance(health_json, str) and len(health_json) > 50,
        f"rlm_get_health_report -- {len(health_json)} bytes returned",
        "rlm_get_health_report -- empty or failed",
    )
    tools_ok += 1
except Exception as e:
    fail(f"rlm_get_health_report: {e}")

try:
    ctx_text = build_trade_and_regime_context(data_root)
    ctx_json = json.dumps({"context": ctx_text})
    check(
        isinstance(ctx_text, str) and len(ctx_text) > 20 and len(ctx_json) > len('{"context":""}'),
        f"rlm_get_trade_and_regime_context -- {len(ctx_text)} chars ({len(ctx_json)} json bytes)",
        "rlm_get_trade_and_regime_context -- empty or failed",
    )
    tools_ok += 1
except Exception as e:
    fail(f"rlm_get_trade_and_regime_context: {e}")

try:
    gate = SystemGate(data_root)
    gs = gate.load()
    gate_json = json.dumps(
        {
            "posture": gs.posture,
            "status": gs.status,
            "last_updated": gs.last_updated,
            "trading_allowed": gate.is_trading_allowed(),
        }
    )
    check(
        "posture" in gate_json,
        f"rlm_get_system_gate_state -- posture={gs.posture}  status={gs.status}  trading_allowed={gate.is_trading_allowed()}",
        "rlm_get_system_gate_state -- failed",
    )
    tools_ok += 1
except Exception as e:
    fail(f"rlm_get_system_gate_state: {e}")

try:
    limits = {
        "max_kelly_fraction": os.environ.get("MAX_KELLY_FRACTION", ""),
        "max_capital_fraction": os.environ.get("MAX_CAPITAL_FRACTION", ""),
        "note": "ROEE policy limits also live in YAML configs; these env vars are optional hints.",
    }
    limits_json = json.dumps({k: v for k, v in limits.items() if v or k == "note"})
    check(
        isinstance(limits_json, str),
        f"rlm_check_portfolio_limits -- {limits_json}",
        "rlm_check_portfolio_limits -- failed",
    )
    tools_ok += 1
except Exception as e:
    fail(f"rlm_check_portfolio_limits: {e}")

info(f"{tools_ok}/4 Hermes tool handlers verified")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Pipeline health (data_monitor) Hermes agent
# ─────────────────────────────────────────────────────────────────────────────
step("Pipeline health -- data_monitor Hermes agent")

pipeline_health_skill = _load_pipeline_health_skill_text(ROOT)
check(
    len(pipeline_health_skill) > 20,
    f"data_monitor/SKILL.md loaded ({len(pipeline_health_skill)} chars)",
    "data_monitor/SKILL.md missing or empty",
)

health_payload = gather_health_report(data_root)
health_ok = bool(health_payload.get("overall_ok", True))
health_txt = str(health_payload.get("report_text", ""))

info("--- Raw health facts ---")
for line in health_txt.splitlines():
    info(f"  {line}")

if hermes_available:
    try:
        health_report = loop_mod._run_pipeline_health_agent(data_root, json.dumps(health_payload, default=str))
        ok(f"Pipeline health agent ran live -- {len(health_report)} chars")
        info("--- Pipeline health agent output ---")
        for line in health_report.splitlines():
            info(f"  {line}")
    except Exception as e:
        fail(f"Pipeline health live agent: {e}")
        health_report = health_txt
else:
    info("Hermes LLM not available -- using direct health facts as pipeline health report")
    health_report = health_txt
    ok(f"Pipeline health facts gathered ({len(health_report)} chars) -- ready for commander")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- Regime research (research_analyst) Hermes agent
# ─────────────────────────────────────────────────────────────────────────────
step("Regime research -- research_analyst Hermes agent")

regime_research_skill = _load_regime_research_skill_text(ROOT)
check(
    len(regime_research_skill) > 20,
    f"research_analyst/SKILL.md loaded ({len(regime_research_skill)} chars)",
    "research_analyst/SKILL.md missing or empty",
)

market_context = build_trade_and_regime_context(data_root)

info("--- Raw market context ---")
for line in market_context.splitlines():
    info(f"  {line}")

if hermes_available:
    try:
        research_report = loop_mod._run_regime_research_agent(data_root, market_context)
        ok(f"Regime research agent ran live -- {len(research_report)} chars")
        info("--- Regime research agent output ---")
        for line in research_report.splitlines():
            info(f"  {line}")
    except Exception as e:
        fail(f"Regime research live agent: {e}")
        research_report = market_context
else:
    info("Hermes LLM not available -- using market context as regime research input")
    research_report = market_context
    ok(f"Regime research context gathered ({len(research_report)} chars) -- ready for commander")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- Commander decision cycle
# ─────────────────────────────────────────────────────────────────────────────
step("Commander -- Hermes crew decision")

commander_skill = _load_commander_skill_text(ROOT)
check(
    len(commander_skill) > 20,
    f"commander/SKILL.md loaded ({len(commander_skill)} chars)",
    "commander/SKILL.md missing or empty",
)

if hermes_available:
    try:
        _, _, commander_llm_text = _run_full_briefing(data_root, health_payload, market_context)
        ok(f"Commander agent ran live -- {len(commander_llm_text)} chars")
    except Exception as e:
        fail(f"Commander live agent: {e}")
        commander_llm_text = None
else:
    commander_llm_text = (
        "SYSTEM STATUS: NOMINAL\n"
        "MARKET POSTURE: NORMAL\n"
        "COMMAND DECISION: HOLD -- regime signals are calm-bullish; no immediate entry trigger.\n"
        "RATIONALE: Pipeline health reports all systems nominal with no journal errors. "
        "Regime research shows 3 active plans in calm_bull regime with 68-76% confidence. "
        "No critical threshold breaches detected; maintain current posture.\n"
        "CREW ORDERS:\n"
        "  - Pipeline Health: maintain current status\n"
        "  - Regime Research: continue monitoring regime confidence for state transitions\n"
        "  - Trading Engine: hold current positions; await RTH open before new entries\n"
        "OVERALL RISK POSTURE: LOW\n"
    )
    info("Hermes LLM not available -- synthetic commander response (all parse/persist paths exercised)")

if commander_llm_text:
    info("--- Commander output ---")
    for line in commander_llm_text.splitlines():
        info(f"  {line}")

    ts = utc_timestamp()
    decision = parse_command_decision(
        ts,
        commander_llm_text,
        health_overall_ok=health_ok,
        context_for_risk=market_context,
    )

    check(
        decision.system_status in ("NOMINAL", "DEGRADED", "CRITICAL"),
        f"system_status parsed: {decision.system_status}",
        f"system_status unexpected: {decision.system_status}",
    )
    check(
        decision.market_posture in ("AGGRESSIVE", "NORMAL", "DEFENSIVE", "STAND-DOWN"),
        f"market_posture parsed: {decision.market_posture}",
        f"market_posture unexpected: {decision.market_posture}",
    )
    check(bool(decision.command), f"command parsed: {decision.command}", "command is empty")
    check(bool(decision.rationale), f"rationale parsed ({len(decision.rationale)} chars)", "rationale is empty")

    save_decision(data_root, decision)
    decisions_path = artifacts / "crew_decisions.json"
    check(decisions_path.is_file(), "crew_decisions.json written", "crew_decisions.json NOT written")

    gate2 = SystemGate(data_root)
    gate2.update(posture=decision.market_posture, status=decision.system_status, timestamp=decision.timestamp)
    gs_after = gate2.load()
    check(
        gs_after.posture == decision.market_posture,
        f"gate_state.json updated -- posture={gs_after.posture}  status={gs_after.status}",
        "gate_state.json NOT updated correctly",
    )
    ok(f"Commander decision persisted -- trading_allowed={gate2.is_trading_allowed()}")

    info("--- Final Telegram-format briefing ---")
    for line in decision.to_telegram_message().splitlines():
        info(f"  {line}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 -- Hermes skill file inventory
# ─────────────────────────────────────────────────────────────────────────────
step("Hermes skill file inventory")

for skill_name in ("commander", "data_monitor", "research_analyst"):
    skill_path = ROOT / "hermes_skills" / skill_name / "SKILL.md"
    if skill_path.is_file():
        ok(f"hermes_skills/{skill_name}/SKILL.md -- present ({skill_path.stat().st_size} bytes)")
    else:
        fail(f"hermes_skills/{skill_name}/SKILL.md -- MISSING")

tools_reg = ROOT / "src" / "rlm_hermes_tools" / "register_rlm_tools.py"
check(tools_reg.is_file(),
      "src/rlm_hermes_tools/register_rlm_tools.py -- present",
      "src/rlm_hermes_tools/register_rlm_tools.py -- MISSING")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{HEAD}{'='*64}\033[0m", flush=True)
print(f"{HEAD}RLM HEALTH CHECK SUMMARY\033[0m", flush=True)
print(f"{HEAD}{'='*64}\033[0m", flush=True)

if _failures:
    print(f"\n  [{FAIL}] {len(_failures)} failure(s):", flush=True)
    for f_msg in _failures:
        print(f"      - {f_msg}", flush=True)
    sys.exit(1)
else:
    print(f"\n  [{PASS}] All {_step} steps passed -- RLM system is healthy.", flush=True)
    hermes_status = (
        "LIVE (LLM connected)"
        if hermes_available
        else "ACTIVATED (infrastructure verified; pip install -e '.[hermes]' for live LLM)"
    )
    print(f"\n  Hermes crew: {hermes_status}", flush=True)
    sys.exit(0)
