#!/usr/bin/env python3
"""
run_crew.py — Hermes-backed RLM crew (health facts + market context + commander).

Usage::

    python scripts/run_crew.py               # continuous loop (default)
    python scripts/run_crew.py --once        # single commander cycle then exit
    python scripts/run_crew.py --health-only # print health JSON/text only (no LLM)
    python scripts/run_crew.py --context-only # print market context only (no LLM)

Requires: ``pip install -e ".[hermes]"`` and a reachable OpenAI-compatible endpoint
(default: local Ollama at ``http://127.0.0.1:11434/v1``).

Environment:
    RLM_ROOT / --root       repo root (default: parent of scripts/)
    RLM_HERMES_BASE_URL     optional explicit OpenAI-compatible base URL
    OPENROUTER_API_KEY      if set (and BASE_URL unset), uses OpenRouter free-tier defaults
    RLM_HERMES_AUTO_GROQ    set 1 with GROQ_API_KEY to use Groq (Groq is never implied from the key alone)
    RLM_HERMES_API_KEY      API key for explicit BASE_URL (default ollama for Ollama)
    RLM_HERMES_MODEL        model id (defaults vary by backend)
    RLM_HERMES_SKIP_MEMORY  1 to disable Hermes persistent memory reads/writes
    RLM_HERMES_TELEGRAM_BOT_TOKEN, RLM_HERMES_TELEGRAM_CHAT_ID
    (legacy fallback: TELEGRAM_BOT_TOKEN, TELEGRAM_NOTIFY_CHAT_ID)
    CREW_HEALTH_INTERVAL, CREW_ANALYSIS_INTERVAL, CREW_BRIEFING_INTERVAL
    RLM_HEALTH_AUTO_RESTART, RLM_HEALTH_RESTART_ALLOW_CREW, RLM_HEALTH_RESTART_COOLDOWN_SEC
        (optional ``SCOTTY_*`` legacy aliases still honored by gather_health_report)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLM Hermes crew")
    p.add_argument("--once", action="store_true", help="Run one commander cycle and exit")
    p.add_argument("--health-only", action="store_true", help="Print gather_health_report JSON and exit")
    p.add_argument("--context-only", action="store_true", help="Print build_trade_and_regime_context and exit")
    p.add_argument("--root", default=str(ROOT), help="Repo root path")
    p.add_argument(
        "--health-interval",
        type=int,
        default=int(os.environ.get("CREW_HEALTH_INTERVAL", "120")),
    )
    p.add_argument(
        "--analysis-interval",
        type=int,
        default=int(os.environ.get("CREW_ANALYSIS_INTERVAL", "300")),
    )
    p.add_argument(
        "--briefing-interval",
        type=int,
        default=int(os.environ.get("CREW_BRIEFING_INTERVAL", "600")),
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    root = Path(args.root).resolve()
    os.environ["RLM_ROOT"] = str(root)

    from rlm.hermes_facts.health import gather_health_report
    from rlm.hermes_facts.market_context import build_trade_and_regime_context

    if args.health_only:
        print(json.dumps(gather_health_report(root), indent=2, default=str))
        return 0
    if args.context_only:
        print(build_trade_and_regime_context(root))
        return 0

    from rlm.hermes_crew.backends import resolve_hermes_backend_tuples
    from rlm.hermes_crew.loop import HermesCrewConfig, run_crew_forever, run_crew_once

    cfg = HermesCrewConfig(
        health_interval=args.health_interval,
        analysis_interval=args.analysis_interval,
        briefing_interval=args.briefing_interval,
    )
    backends = resolve_hermes_backend_tuples()
    primary = backends[0]
    print(
        f"[Crew] Hermes backends — root={root} primary={primary[0]!r} model={primary[2]!r}"
        + (f" fallbacks={len(backends) - 1}" if len(backends) > 1 else ""),
        flush=True,
    )
    if args.once:
        decision = run_crew_once(root, cfg)
        print("\n--- Commander decision ---")
        print(decision.to_telegram_message())
        return 0
    run_crew_forever(root, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
