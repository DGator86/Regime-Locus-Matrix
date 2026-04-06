"""Broker execution helpers (IBKR combo / option orders)."""

from __future__ import annotations

from rlm.execution.ibkr_combo_orders import (
    IBKRLegAction,
    IBKROptionLegSpec,
    assert_paper_or_live_acknowledged,
    assert_paper_trading_port,
    expiry_iso_to_ib,
    legs_from_ibkr_combo_spec,
    load_ibkr_order_socket_config,
    place_options_combo_limit_order,
    place_options_combo_market_order,
    place_options_combo_order,
    resolve_option_contract,
    reverse_legs_for_close,
)
from rlm.execution.risk_targets import SpreadExitThresholds, build_spread_exit_thresholds, trailing_stop_from_peak

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
