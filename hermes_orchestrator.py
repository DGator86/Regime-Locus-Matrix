#!/usr/bin/env python3
"""CLI entry for the Hermes-backed RLM crew (optional; prefer ``scripts/run_crew.py``)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="RLM Hermes crew orchestrator")
    p.add_argument("--once", action="store_true", help="Single commander cycle")
    p.add_argument("--root", default=str(ROOT), help="Repo root")
    args = p.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.append(str(root))
    os.environ.setdefault("RLM_ROOT", str(root))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    from rlm.hermes_crew.loop import run_crew_forever, run_crew_once

    if args.once:
        d = run_crew_once(root)
        print(d.to_telegram_message())
        return 0
    run_crew_forever(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
