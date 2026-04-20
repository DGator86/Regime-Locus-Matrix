"""Thin wrapper — delegates to ``rlm forecast``.

Kept for backwards compatibility. New usage: ``rlm forecast [options]``
"""

import sys
from rlm.cli.forecast import main

if __name__ == "__main__":
    main()
