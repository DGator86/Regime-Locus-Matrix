"""Host-watchdog restart policy for market-hours trading units.

Kept separate from ``scripts/rlm_enterprise_watchdog.py`` so unit tests do not need
``psutil`` / Telegram deps, and so import works under ``systemd_exec_python``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from rlm.utils.market_hours import is_market_service_window_open

# Trading-heavy units started/stopped by rlm-market-open/close timers. Host watchdog must not
# undo market-close by restarting them overnight/weekend (bootstrap WATCHED_SERVICES includes
# rlm-master-trader for in-window crash recovery only).
MARKET_HOURS_RESTART_SERVICES = frozenset(
    {
        "rlm-master-trader",
        "regime-locus-master",
        "rlm-master-telegram",
        "rlm-challenge-loop",
    }
)


def watched_service_base_name(svc: str) -> str:
    name = str(svc or "").strip()
    if name.endswith(".service"):
        name = name[: -len(".service")]
    return name


def should_auto_restart_watched_service(
    svc: str,
    *,
    _override: Optional[datetime] = None,
) -> bool:
    """False for trading-heavy units outside the NYSE market-open/close service window.

    ``bootstrap.sh`` / ``enterprise.env`` list ``rlm-master-trader`` in ``WATCHED_SERVICES`` so
    the host watchdog can recover crashes *during* the trading day. Without this gate, the
    16:30 ``rlm-market-hours-stop.sh`` stop is undone within one poll interval.
    """
    base = watched_service_base_name(svc)
    if base not in MARKET_HOURS_RESTART_SERVICES:
        return True
    return bool(is_market_service_window_open(_override=_override))
