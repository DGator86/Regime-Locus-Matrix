"""Thin wrapper — delegates to ``rlm backtest --walkforward``.

Kept for backwards compatibility. New usage: ``rlm backtest --walkforward [options]``
"""

import sys
sys.argv = [sys.argv[0], "--walkforward", *sys.argv[1:]]

from rlm.cli.backtest import main

if __name__ == "__main__":
    main()
