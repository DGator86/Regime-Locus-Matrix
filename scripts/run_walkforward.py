"""Run walk-forward validation across all universe symbols that have data on disk.

Symbols without a local bars file are silently skipped.

Usage (from repo root, package installed):

    python scripts/run_walkforward.py [options]
    python scripts/run_walkforward.py --symbols SPY,QQQ
    python scripts/run_walkforward.py --symbol SPY        # single symbol

Any extra flags are forwarded to ``rlm backtest`` (e.g. --regime hmm).
"""

import sys

# Inject --walkforward --universe before any user-supplied flags so that
# the full LIQUID_UNIVERSE is iterated by default.  Individual --symbol /
# --symbols overrides still work because the CLI group is mutually exclusive.
sys.argv = [sys.argv[0], "--walkforward", "--universe", *sys.argv[1:]]

from rlm.cli.backtest import main  # noqa: E402

if __name__ == "__main__":
    main()
