"""Canonical option exit signals used across monitoring and notifications."""

from __future__ import annotations

EXIT_SIGNALS = frozenset(
    {
        "take_profit",
        "hard_stop",
        "trailing_stop",
        "expiry_force_close",
        "time_stop",
        "max_loss_stop",
    }
)

