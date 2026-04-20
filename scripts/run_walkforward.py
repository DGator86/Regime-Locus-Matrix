"""Legacy wrapper — backwards compatibility shim for ``rlm backtest --walkforward``.

Requires the package to be installed:

    pip install -e .

Preferred usage (works from any directory after install):

    rlm backtest --symbol SPY --walkforward [options]

This wrapper will be removed in a future release.
"""

import sys

sys.argv = [sys.argv[0], "--walkforward", *sys.argv[1:]]

from rlm.cli.backtest import main

if __name__ == "__main__":
    main()
