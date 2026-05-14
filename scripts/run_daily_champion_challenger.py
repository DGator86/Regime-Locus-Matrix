#!/usr/bin/env python3
"""Daily champion/challenger automation with a manual promotion gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--approve-promotion", action="store_true", help="Allow live-model promotion.")
    p.add_argument("--no-vix", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    nv = ["--no-vix"] if args.no_vix else []
    run([sys.executable, "scripts/build_features.py", "--symbol", args.symbol, *nv])
    run([sys.executable, "scripts/calibrate_regime_models.py", "--symbol", args.symbol, "--no-promote", *nv])
    if args.approve_promotion or os.environ.get("RLM_PROMOTION_APPROVED", "").strip() == "1":
        run([sys.executable, "scripts/calibrate_regime_models.py", "--symbol", args.symbol, *nv])
    else:
        print("Promotion skipped. Re-run with --approve-promotion or set RLM_PROMOTION_APPROVED=1.")


if __name__ == "__main__":
    main()
