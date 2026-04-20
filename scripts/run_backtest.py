"""Legacy wrapper — backwards compatibility shim for ``rlm backtest``.

Requires the package to be installed:

    pip install -e .

Preferred usage (works from any directory after install):

    rlm backtest --symbol SPY [options]

This wrapper will be removed in a future release.
"""

from rlm.cli.backtest import main

if __name__ == "__main__":
    main()
