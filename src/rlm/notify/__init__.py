"""Optional notification helpers (e.g. Telegram) for RLM file-driven state."""

from __future__ import annotations

from typing import Any

_EXPORT_MODULES = {
    "build_balances_text": "rlm.notify.telegram_rlm",
    "build_pnl_text": "rlm.notify.telegram_rlm",
    "build_session_brief_text": "rlm.notify.telegram_rlm",
    "build_status_brief": "rlm.notify.telegram_rlm",
    "build_universe_and_positions": "rlm.notify.telegram_rlm",
    "build_universe_report": "rlm.notify.telegram_rlm",
    "default_paths": "rlm.notify.telegram_rlm",
    "load_notify_state": "rlm.notify.telegram_rlm",
    "notification_cycle": "rlm.notify.telegram_rlm",
    "save_notify_state": "rlm.notify.telegram_rlm",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(_EXPORT_MODULES[name]), name)
    globals()[name] = value
    return value
