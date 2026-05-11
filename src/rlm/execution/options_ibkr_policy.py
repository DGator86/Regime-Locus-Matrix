"""Policy: IBKR connectivity is for **equities**; option combo orders are opt-in only.

Large-account options and the PDT challenge are measured on disk (``trade_log``, challenge
state).  Transmitting opens/closes for multi-leg options requires ``RLM_ALLOW_IBKR_OPTIONS=1``.
"""

from __future__ import annotations

import os
import sys


def ibkr_option_orders_allowed() -> bool:
    v = (os.environ.get("RLM_ALLOW_IBKR_OPTIONS") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def exit_if_ibkr_option_orders_disallowed(context: str) -> None:
    """``sys.exit(2)`` with a clear message when option combo transmission is not allowed."""
    if ibkr_option_orders_allowed():
        return
    print(
        "RLM policy: IBKR is reserved for equities. Option combo orders are not transmitted.\n"
        f"  ({context})\n"
        "  Large-account options and PDT challenge are tracked locally only.\n"
        "  To override (maintainers only): set RLM_ALLOW_IBKR_OPTIONS=1",
        file=sys.stderr,
    )
    raise SystemExit(2)
