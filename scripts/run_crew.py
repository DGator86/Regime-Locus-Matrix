#!/usr/bin/env python3
"""
run_crew.py — launch the Starfleet AI crew (Scotty / Spock / Kirk).

Usage::

    python scripts/run_crew.py               # continuous loop (default)
    python scripts/run_crew.py --once        # single cycle then exit
    python scripts/run_crew.py --scotty      # health check only
    python scripts/run_crew.py --spock       # market analysis only
    python scripts/run_crew.py --backend ollama --model mistral   # use local Ollama

Environment (.env):
    GROQ_API_KEY            Free key from console.groq.com  (backend=groq)
    LLM_BACKEND             groq | ollama  (default: groq)
    LLM_MODEL               override model name
    OLLAMA_HOST             Ollama server URL (default: http://localhost:11434)
    TELEGRAM_BOT_TOKEN      existing RLM Telegram bot token
    TELEGRAM_NOTIFY_CHAT_ID chat to post crew messages into
    CREW_HEALTH_INTERVAL    seconds between Scotty checks   (default 120)
    CREW_ANALYSIS_INTERVAL  seconds between Spock analyses  (default 300)
    CREW_BRIEFING_INTERVAL  seconds between Kirk briefings  (default 600)
    CREW_SERVICES           comma-separated systemd unit names to monitor
    SCOTTY_AUTO_RESTART     1 = try systemctl restart on loaded-but-inactive watched units (default 1)
    SCOTTY_RESTART_COOLDOWN_SEC  min seconds between restarts per unit (default 180)
    SCOTTY_RESTART_ALLOW_CREW set 1 to allow restarting regime-locus-crew (unsafe from run_crew)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load .env before importing agents (they read env vars at import time)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLM Starfleet AI Crew")
    p.add_argument("--once", action="store_true", help="Run one full cycle and exit")
    p.add_argument("--scotty", action="store_true", help="Scotty health check only")
    p.add_argument("--spock", action="store_true", help="Spock market analysis only")
    p.add_argument("--backend", choices=["groq", "ollama"], help="LLM backend override")
    p.add_argument("--model", help="LLM model name override")
    p.add_argument("--root", default=str(ROOT), help="Repo root path")
    p.add_argument(
        "--health-interval",
        type=int,
        default=int(os.environ.get("CREW_HEALTH_INTERVAL", "120")),
        help="Scotty check interval in seconds",
    )
    p.add_argument(
        "--analysis-interval",
        type=int,
        default=int(os.environ.get("CREW_ANALYSIS_INTERVAL", "300")),
        help="Spock analysis interval in seconds",
    )
    p.add_argument(
        "--briefing-interval",
        type=int,
        default=int(os.environ.get("CREW_BRIEFING_INTERVAL", "600")),
        help="Kirk briefing interval in seconds",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    root = Path(args.root).resolve()

    # Apply CLI overrides to env so LLMConfig picks them up
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    from rlm.agents.base import LLMClient, LLMConfig
    from rlm.agents.crew import CrewConfig, StarfleetCrew
    from rlm.agents.scotty import ScottyAgent
    from rlm.agents.spock import SpockAgent

    llm_cfg = LLMConfig.from_env()
    print(
        f"[Crew] Starting — backend={llm_cfg.backend} model={llm_cfg.model} root={root}",
        flush=True,
    )

    # Scotty-only mode
    if args.scotty:
        agent = ScottyAgent(root, LLMClient(llm_cfg))
        report, diagnosis = agent.check()
        print(report.to_text())
        print("\n--- Scotty's Diagnosis ---")
        print(diagnosis)
        return 0

    # Spock-only mode
    if args.spock:
        agent = SpockAgent(root, LLMClient(llm_cfg))
        briefing = agent.analyse()
        print(briefing.context_snapshot)
        print("\n--- Spock's Briefing ---")
        print(briefing.llm_text)
        return 0

    # Full crew
    crew_cfg = CrewConfig(
        health_interval=args.health_interval,
        analysis_interval=args.analysis_interval,
        briefing_interval=args.briefing_interval,
    )
    crew = StarfleetCrew(root=root, config=crew_cfg, llm=LLMClient(llm_cfg))

    if args.once:
        decision = crew.run_once()
        print("\n--- Kirk's Decision ---")
        print(decision.to_telegram_message())
        return 0

    # Continuous loop (normal service mode)
    crew.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
