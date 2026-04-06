#!/usr/bin/env python3
"""
Run the **full stack** from repo root in order:

1. ``run_universe_options_pipeline.py`` — IBKR + factors + forecast + ROEE + Massive chain match + risk plan JSON
2. Optional: ``ibkr_paper_trade_from_plans.py`` — **paper** opening combos (``IBKR_PORT`` 7497 / 4002)
3. ``monitor_active_trade_plans.py`` — Massive marks vs stops; optional ``--paper-close`` **MKT** exits

Examples::

    python scripts/run_everything.py
    python scripts/run_everything.py --full-paper --interval 120
    python scripts/run_everything.py --paper-trade --paper-close --follow --interval 120
    python scripts/run_everything.py --pipeline-args "--top 2 --no-vix"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    return int(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        default="data/processed/universe_trade_plans.json",
        help="Plans JSON path (relative to repo root unless absolute)",
    )
    ap.add_argument(
        "--pipeline-args",
        default="",
        help="Extra args for run_universe_options_pipeline.py (quoted), e.g. '--top 3 --no-vix'",
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="After pipeline, run monitor in a loop (see --interval) instead of --once",
    )
    ap.add_argument("--interval", type=float, default=120.0, help="Monitor poll seconds when --follow")
    ap.add_argument("--skip-monitor", action="store_true", help="Only run the universe options pipeline")
    ap.add_argument("--skip-pipeline", action="store_true", help="Only run monitor (plans file must exist)")
    ap.add_argument(
        "--paper-trade",
        action="store_true",
        help="After pipeline, place opening LMT combos from plans (paper IBKR only)",
    )
    ap.add_argument("--paper-trade-max", type=int, default=10, help="Cap opening orders")
    ap.add_argument("--paper-dry-run", action="store_true", help="Log openings only (no IBKR transmit)")
    ap.add_argument(
        "--paper-close",
        action="store_true",
        help="Monitor transmits MKT closes on exit signals (paper IBKR only)",
    )
    ap.add_argument("--paper-close-dry-run", action="store_true", help="Log closes only")
    ap.add_argument(
        "--full-paper",
        action="store_true",
        help="Shorthand: --paper-trade --paper-close --follow (continuous monitor + paper in/out)",
    )
    args = ap.parse_args()

    if args.full_paper:
        args.paper_trade = True
        args.paper_close = True
        args.follow = True

    py = sys.executable
    plans = args.out

    if not args.skip_pipeline:
        cmd = [py, str(ROOT / "scripts" / "run_universe_options_pipeline.py"), "--out", plans]
        if args.pipeline_args.strip():
            cmd.extend(shlex.split(args.pipeline_args))
        rc = _run(cmd)
        if rc != 0:
            return rc

    if args.paper_trade:
        pcmd = [
            py,
            str(ROOT / "scripts" / "ibkr_paper_trade_from_plans.py"),
            "--plans",
            plans,
            "--max",
            str(args.paper_trade_max),
        ]
        if args.paper_dry_run:
            pcmd.append("--dry-run")
        rc = _run(pcmd)
        if rc != 0:
            return rc

    if args.skip_monitor:
        return 0

    mcmd = [
        py,
        str(ROOT / "scripts" / "monitor_active_trade_plans.py"),
        "--plans",
        plans,
    ]
    if args.follow:
        mcmd.extend(["--interval", str(args.interval)])
    else:
        mcmd.append("--once")
    if args.paper_close:
        mcmd.append("--paper-close")
    if args.paper_close_dry_run:
        mcmd.append("--paper-close-dry-run")
    return _run(mcmd)


if __name__ == "__main__":
    raise SystemExit(main())
