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
# the full LIQUID_UNIVERSE is iterated by default.
user_args = sys.argv[1:]
has_explicit_symbol = any(
    arg == "--symbol"
    or arg.startswith("--symbol=")
    or arg == "--symbols"
    or arg.startswith("--symbols=")
    for arg in user_args
)
default_selector = [] if has_explicit_symbol else ["--universe"]
sys.argv = [sys.argv[0], "--walkforward", *default_selector, *user_args]

from rlm.cli.backtest import main  # noqa: E402

if __name__ == "__main__":
    main()
