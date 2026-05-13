#!/usr/bin/env python3
"""
Single entrypoint for the full automated dual-book stack.

Runs the **equity-primary** mode by default:

1. ``run_universe_options_pipeline.py`` — IBKR bars, factor/forecast pipeline, ROEE, Massive chain match
2. ``ibkr_paper_trade_from_plans.py``  — option opens **dry-run** only (``run_everything`` enforces; no IBKR combos)
3. ``ibkr_equity_paper_trade.py``      — real IBKR **paper** stock BUY / SELL from regime direction
4. ``monitor_active_trade_plans.py``   — Massive marks vs stops; **local** ``trade_log.csv`` (no IBKR option closes)
5. ``rlm challenge --run`` (when not ``--skip-challenge``) — **PDT** $1K→$25K options dry-run book under ``data/challenge/``

IBKR paper accounts support equity trading only; options are tracked hypothetically so you can
measure regime-signal quality against both actual equity execution and simulated options P&L.

Examples::

    python scripts/run_master.py
    python scripts/run_master.py --pipeline-args "--workers 4"
    python scripts/run_master.py --interval 30 --rescan-interval 600
    python scripts/run_master.py --equity-position-usd 5000 --equity-stop-pct 3
    python scripts/run_master.py --equity-dry-run   # log equity signals without IBKR orders
    python scripts/run_master.py --skip-challenge   # omit PDT $1K→$25K sleeve (default: included)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_everything.py"),
        "--master",
        "--with-equity",
        "--with-challenge",
        *sys.argv[1:],
    ]
    return int(subprocess.run(cmd, cwd=str(ROOT)).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
