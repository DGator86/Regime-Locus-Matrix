"""Legacy wrapper — backwards compatibility shim for ``rlm backtest --walkforward``.

Requires the package to be installed:

    pip install -e .

Preferred usage (works from any directory after install):

    rlm backtest --symbol SPY --walkforward [options]
    rlm backtest --universe --walkforward

With no symbol flags, this script prepends ``--walkforward`` and ``--universe`` so
a bare ``python scripts/run_walkforward.py`` runs walk-forward for every ticker in
:data:`~rlm.data.liquidity_universe.EXPANDED_LIQUID_UNIVERSE`.

This wrapper will be removed in a future release.
"""

import sys


def _has_symbol_args(argv: list[str]) -> bool:
    for i, a in enumerate(argv):
        if a in ("--symbol", "--symbols", "--universe"):
            return True
        if a.startswith("--symbol=") or a.startswith("--symbols="):
            return True
    return False


_rest = list(sys.argv[1:])
if not _has_symbol_args(_rest):
    _rest = ["--universe", *_rest]
sys.argv = [sys.argv[0], "--walkforward", *_rest]

from rlm.cli.backtest import main

if __name__ == "__main__":
    main()
