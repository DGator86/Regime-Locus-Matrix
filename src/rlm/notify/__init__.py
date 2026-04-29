"""Optional notification helpers (e.g. Telegram) for RLM file-driven state."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "build_balances_text",
    "build_pnl_text",
    "build_status_brief",
    "build_universe_and_positions",
    "build_universe_report",
    "default_paths",
    "load_notify_state",
    "notification_cycle",
    "save_notify_state",
]

_LAZY: dict[str, tuple[str, str]] = {
    name: ("rlm.notify.telegram_rlm", name) for name in __all__
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
