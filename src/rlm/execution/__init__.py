"""Broker execution helpers (IBKR combo / option orders).

Public helpers are resolved lazily so importing lightweight submodules under
``rlm.execution`` does not pull in pandas/IBKR dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "IBKRLegAction": "rlm.execution.ibkr_combo_orders",
    "IBKROptionLegSpec": "rlm.execution.ibkr_combo_orders",
    "SpreadExitThresholds": "rlm.execution.risk_targets",
    "assert_paper_or_live_acknowledged": "rlm.execution.ibkr_combo_orders",
    "assert_paper_trading_port": "rlm.execution.ibkr_combo_orders",
    "build_spread_exit_thresholds": "rlm.execution.risk_targets",
    "expiry_iso_to_ib": "rlm.execution.ibkr_combo_orders",
    "legs_from_ibkr_combo_spec": "rlm.execution.ibkr_combo_orders",
    "load_ibkr_order_socket_config": "rlm.execution.ibkr_combo_orders",
    "place_options_combo_limit_order": "rlm.execution.ibkr_combo_orders",
    "place_options_combo_market_order": "rlm.execution.ibkr_combo_orders",
    "place_options_combo_order": "rlm.execution.ibkr_combo_orders",
    "resolve_option_contract": "rlm.execution.ibkr_combo_orders",
    "reverse_legs_for_close": "rlm.execution.ibkr_combo_orders",
    "trailing_stop_from_peak": "rlm.execution.risk_targets",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
