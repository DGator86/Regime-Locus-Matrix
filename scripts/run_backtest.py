"""Thin wrapper — delegates to ``rlm backtest``.

Kept for backwards compatibility. New usage: ``rlm backtest [options]``
"""

import sys
from rlm.cli.backtest import main

if __name__ == "__main__":
    main()
