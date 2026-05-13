"""Policy: RLM never transmits option or multi-leg combo orders via Interactive Brokers."""

from __future__ import annotations

import sys


def exit_ibkr_option_combo_blocked(context: str) -> None:
    """Print a clear message and ``sys.exit(2)`` — option combos are not brokered through IBKR."""
    sys.stderr.write(
        "RLM policy: options are not executed via Interactive Brokers in this stack.\n"
        f"  ({context})\n"
        "  Track options locally (trade_log, challenge); use IBKR only for the equity leg "
        "(ibkr_equity_paper_trade.py) if enabled.\n"
    )
    raise SystemExit(2)
