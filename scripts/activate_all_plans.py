#!/usr/bin/env python3
"""Unified RLM activation coordinator.

Thin wrapper around ``rlm activate`` — identical flags, same behaviour.
Useful for launching directly from a shell without the ``rlm`` CLI installed.

Phase 1 — Parallel bulk data import for the full universe.
Phase 2 — Three plans start simultaneously:
  • Equity plan      IBKR paper equity BUY/SELL from regime signals
  • Options plan     Universe options pipeline + marks monitor (continuous)
  • Challenge plan   $1K→$25K aggressive session for SPY + QQQ

Examples::

    python scripts/activate_all_plans.py
    python scripts/activate_all_plans.py --no-ingest
    python scripts/activate_all_plans.py --equity-dry-run
    python scripts/activate_all_plans.py --no-challenge
    python scripts/activate_all_plans.py --ingest-workers 6
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Delegate entirely to the CLI module so there is one source of truth.
from rlm.cli.activate import main  # noqa: E402

if __name__ == "__main__":
    main()
