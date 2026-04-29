"""Broker execution helpers (IBKR combo / option orders)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "IBKRLegAction",
    "IBKROptionLegSpec",
    "SpreadExitThresholds",
    "assert_paper_or_live_acknowledged",
    "assert_paper_trading_port",
    "build_spread_exit_thresholds",
    "expiry_iso_to_ib",
    "legs_from_ibkr_combo_spec",
    "load_ibkr_order_socket_config",
    "place_options_combo_limit_order",
    "place_options_combo_market_order",
    "place_options_combo_order",
    "resolve_option_contract",
    "reverse_legs_for_close",
    "trailing_stop_from_peak",
]

_LAZY: dict[str, tuple[str, str]] = {
    "IBKRLegAction": ("rlm.execution.ibkr_combo_orders", "IBKRLegAction"),
    "IBKROptionLegSpec": ("rlm.execution.ibkr_combo_orders", "IBKROptionLegSpec"),
    "SpreadExitThresholds": ("rlm.execution.risk_targets", "SpreadExitThresholds"),
    "assert_paper_or_live_acknowledged": (
        "rlm.execution.ibkr_combo_orders",
        "assert_paper_or_live_acknowledged",
    ),
    "assert_paper_trading_port": ("rlm.execution.ibkr_combo_orders", "assert_paper_trading_port"),
    "build_spread_exit_thresholds": ("rlm.execution.risk_targets", "build_spread_exit_thresholds"),
    "expiry_iso_to_ib": ("rlm.execution.ibkr_combo_orders", "expiry_iso_to_ib"),
    "legs_from_ibkr_combo_spec": ("rlm.execution.ibkr_combo_orders", "legs_from_ibkr_combo_spec"),
    "load_ibkr_order_socket_config": ("rlm.execution.ibkr_combo_orders", "load_ibkr_order_socket_config"),
    "place_options_combo_limit_order": ("rlm.execution.ibkr_combo_orders", "place_options_combo_limit_order"),
    "place_options_combo_market_order": ("rlm.execution.ibkr_combo_orders", "place_options_combo_market_order"),
    "place_options_combo_order": ("rlm.execution.ibkr_combo_orders", "place_options_combo_order"),
    "resolve_option_contract": ("rlm.execution.ibkr_combo_orders", "resolve_option_contract"),
    "reverse_legs_for_close": ("rlm.execution.ibkr_combo_orders", "reverse_legs_for_close"),
    "trailing_stop_from_peak": ("rlm.execution.risk_targets", "trailing_stop_from_peak"),
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
