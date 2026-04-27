#!/usr/bin/env python3
"""Post-session performance check: trigger nightly re-optimization when win rate degrades.

Reads ``data/processed/trade_log.csv`` (closed rows only), computes a rolling
win rate over the last ``--lookback`` closed trades, and fires
``run_nightly_hyperparam_opt.py`` when the rate falls below ``--warn-threshold``.
If it falls below ``--critical-threshold`` it also re-runs
``calibrate_regime_models.py`` to promote a new champion regime model.

Designed to be called once per session, after ``monitor_active_trade_plans.py``
finishes (i.e. after market close).  The master service runs this automatically;
you can also call it manually at any time.

Examples::

    python scripts/check_performance_and_retune.py
    python scripts/check_performance_and_retune.py --lookback 30 --warn-threshold 0.38
    python scripts/check_performance_and_retune.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TRADE_LOG = ROOT / "data" / "processed" / "trade_log.csv"
NIGHTLY_SCRIPT = ROOT / "scripts" / "run_nightly_hyperparam_opt.py"
CALIBRATE_SCRIPT = ROOT / "scripts" / "calibrate_regime_models.py"

# Defaults
DEFAULT_LOOKBACK = 20          # closed trades to evaluate
DEFAULT_WARN_THRESHOLD = 0.40  # trigger nightly opt below this win rate
DEFAULT_CRITICAL_THRESHOLD = 0.30  # also trigger regime re-calibration below this


def _read_closed_pnl(path: Path, lookback: int) -> list[float]:
    """Return PnL values for the last ``lookback`` closed trades (closed=1)."""
    if not path.is_file():
        return []
    rows: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("closed", "0")).strip() == "1":
                try:
                    rows.append(float(r.get("unrealized_pnl", 0) or 0))
                except (TypeError, ValueError):
                    rows.append(0.0)
    return rows[-lookback:]


def _win_rate(pnls: list[float]) -> float:
    if not pnls:
        return float("nan")
    return sum(1 for p in pnls if p > 0) / len(pnls)


def _run(cmd: list[str], dry_run: bool) -> int:
    print(f"  {'[dry-run] ' if dry_run else ''}$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                    help=f"Closed-trade window to evaluate (default {DEFAULT_LOOKBACK})")
    ap.add_argument("--warn-threshold", type=float, default=DEFAULT_WARN_THRESHOLD,
                    help=f"Win rate below this fires nightly opt (default {DEFAULT_WARN_THRESHOLD})")
    ap.add_argument("--critical-threshold", type=float, default=DEFAULT_CRITICAL_THRESHOLD,
                    help=f"Win rate below this also fires regime re-calibration (default {DEFAULT_CRITICAL_THRESHOLD})")
    ap.add_argument("--nightly-trials", type=int, default=40,
                    help="Optuna trials for nightly opt (default 40)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands that would run without executing them")
    args = ap.parse_args()

    pnls = _read_closed_pnl(TRADE_LOG, args.lookback)
    if not pnls:
        print(f"check_performance: no closed trades found in {TRADE_LOG} — skipping.", flush=True)
        return 0

    wr = _win_rate(pnls)
    n = len(pnls)
    print(
        f"check_performance: last {n} closed trades — win rate {wr:.1%}  "
        f"(warn<{args.warn_threshold:.0%} critical<{args.critical_threshold:.0%})",
        flush=True,
    )

    if wr >= args.warn_threshold:
        print("check_performance: win rate acceptable — no re-tuning needed.", flush=True)
        return 0

    print(
        f"check_performance: win rate {wr:.1%} below warn threshold {args.warn_threshold:.0%} "
        f"— triggering nightly hyperparameter re-optimization.",
        flush=True,
    )
    rc = _run(
        [sys.executable, str(NIGHTLY_SCRIPT), "--trials", str(args.nightly_trials)],
        args.dry_run,
    )
    if rc != 0:
        print(f"check_performance: nightly opt exited {rc}", file=sys.stderr, flush=True)

    if wr < args.critical_threshold:
        print(
            f"check_performance: win rate {wr:.1%} below critical threshold "
            f"{args.critical_threshold:.0%} — also triggering regime re-calibration.",
            flush=True,
        )
        rc2 = _run(
            [sys.executable, str(CALIBRATE_SCRIPT), "--promote"],
            args.dry_run,
        )
        if rc2 != 0:
            print(f"check_performance: calibration exited {rc2}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
