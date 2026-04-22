#!/usr/bin/env python3
"""Bounded universe/options pipeline run for systemd pre-open / post-close timers.

Writes a separate JSON (default ``data/processed/session_brief.json``) so the main
``universe_trade_plans.json`` used by the master loop is not overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("preopen", "postclose"),
        default="preopen",
        help="Label for logs only (same pipeline either way)",
    )
    p.add_argument("--top", type=int, default=8, help="Cap symbols (passed to universe pipeline)")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/processed/session_brief.json",
        help="Output JSON path (relative to repo root unless absolute)",
    )
    args = p.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_universe_options_pipeline.py"),
        "--no-vix",
        "--top",
        str(int(args.top)),
        "--out",
        str(out),
    ]
    print(f"[session-brief] phase={args.phase} -> {' '.join(cmd)}", flush=True)
    return int(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
