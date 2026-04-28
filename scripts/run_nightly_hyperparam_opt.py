#!/usr/bin/env python3
"""Run multi-timeframe nightly hyperparameter search and write overlay JSON for live runs.

Writes ``data/processed/live_nightly_hyperparams.json`` (best Optuna trial params).
The universe pipeline merges this into :class:`~rlm.forecasting.live_model.LiveRegimeModelConfig`
via :func:`~rlm.forecasting.live_model.apply_nightly_hyperparam_overlay` when present.

Schedule this after the US close (for example 17:30 America/New_York) so bars reflect the session;
use a cron job or systemd timer on the host if you run unattended.

Examples::

    python3 scripts/run_nightly_hyperparam_opt.py
    python3 scripts/run_nightly_hyperparam_opt.py --trials 20 --symbols SPY,QQQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.optimization.nightly import NightlyMTFOptimizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Optuna trials (default: 40; study also stops after 1h timeout)",
    )
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (default: built-in liquid basket)",
    )
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    best = NightlyMTFOptimizer.run(symbols=symbols, trials=max(1, int(args.trials)))
    print(best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
