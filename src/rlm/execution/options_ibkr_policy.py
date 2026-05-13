"""Policy: IBKR is for **equities** only. Multi-leg option combos are never transmitted.

Large-account options and the PDT challenge are tracked on disk (``trade_log``, challenge state).
There is no env override — this is a hard product invariant.
"""

from __future__ import annotations

import sys


def exit_ibkr_option_combo_blocked(context: str) -> None:
    """Print a clear message and ``sys.exit(2)`` — option combo orders are not supported on IBKR."""
    print(
        "RLM policy: IBKR is equities-only. Multi-leg option combos are not transmitted.\n"
        f"  ({context})\n"
        "  Track options locally (trade_log, challenge); execute stocks via ibkr_equity_paper_trade.py.",
        file=sys.stderr,
    )
    raise SystemExit(2)
