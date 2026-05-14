"""Interactive Brokers socket helpers (paper/live port checks, client id).

**Options:** RLM does **not** send option or multi-leg combo orders through IBKR.
Vendor-neutral combo JSON lives in :mod:`rlm.execution.combo_spec`. Functions
named ``place_options_combo_*`` / ``resolve_option_contract`` remain as explicit
:exc:`RuntimeError` stubs so any stale caller fails fast.
"""

from __future__ import annotations

import os
from typing import Any, Literal, NoReturn

from dotenv import load_dotenv

from rlm.execution.combo_spec import (
    ComboLegAction,
    OptionComboLeg,
    expiry_iso_to_yyyymmdd,
    legs_from_combo_spec,
    option_type_to_right,
    reverse_combo_legs,
    side_to_combo_action,
)

IBKRLegAction = ComboLegAction
IBKROptionLegSpec = OptionComboLeg
IBKRAlgoStrategy = Literal["NONE", "ADAPTIVE", "VWAP"]

IBKR_LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})
IBKR_PAPER_PORTS: frozenset[int] = frozenset({7497, 4002, 4004})

# Backward-compatible aliases (new code should import from combo_spec).
expiry_iso_to_ib = expiry_iso_to_yyyymmdd
roee_side_to_ib_action = side_to_combo_action
option_type_to_ib_right = option_type_to_right
legs_from_ibkr_combo_spec = legs_from_combo_spec
reverse_legs_for_close = reverse_combo_legs


def build_ibkr_algo_fields(
    *,
    strategy: IBKRAlgoStrategy = "NONE",
    adaptive_priority: Literal["Urgent", "Normal", "Patient"] = "Normal",
    vwap_max_pct_vol: float = 0.1,
    vwap_start_time: str | None = None,
    vwap_end_time: str | None = None,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Build ``order.algoStrategy`` + ``order.algoParams`` values (legacy / tests)."""
    s = str(strategy).upper().strip()
    if s == "NONE":
        return None, []
    if s == "ADAPTIVE":
        return "Adaptive", [("adaptivePriority", str(adaptive_priority))]
    if s == "VWAP":
        params: list[tuple[str, str]] = [("maxPctVol", f"{float(vwap_max_pct_vol):.4f}")]
        if vwap_start_time:
            params.append(("startTime", str(vwap_start_time)))
        if vwap_end_time:
            params.append(("endTime", str(vwap_end_time)))
        return "Vwap", params
    raise ValueError(f"Unknown algo strategy: {strategy}")


def load_ibkr_order_socket_config() -> tuple[str, int, int]:
    """``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_ORDER_CLIENT_ID`` (else ``IBKR_CLIENT_ID``, else 2)."""
    load_dotenv()
    host = (os.environ.get("IBKR_HOST") or "127.0.0.1").strip()
    port = int(os.environ.get("IBKR_PORT") or "7497")
    cid_raw = os.environ.get("IBKR_ORDER_CLIENT_ID") or os.environ.get("IBKR_CLIENT_ID") or "2"
    client_id = int(cid_raw)
    return host, port, client_id


def assert_paper_or_live_acknowledged(port: int, *, acknowledge_live: bool) -> None:
    if port in IBKR_LIVE_PORTS and not acknowledge_live:
        raise ValueError(
            f"Refusing IBKR orders on live port {port} without acknowledge_live=True "
            f"(use paper {sorted(IBKR_PAPER_PORTS)} or pass the explicit acknowledgement)."
        )


def assert_paper_trading_port(port: int) -> None:
    """Refuse automated *paper* stack on live sockets (catch mis-set ``IBKR_PORT``)."""
    if port not in IBKR_PAPER_PORTS:
        raise ValueError(
            f"Automated paper trading requires IBKR_PORT in {sorted(IBKR_PAPER_PORTS)}; got {port}. "
            "Set paper TWS (7497), Gateway (4002), or host-mapped Gateway (4004)."
        )


def _option_orders_disabled() -> NoReturn:
    raise RuntimeError(
        "RLM does not transmit multi-leg option orders via Interactive Brokers. "
        "Track options in trade_log / external brokers; use IBKR only for equities in this stack."
    )


def resolve_option_contract(*args: Any, **kwargs: Any) -> NoReturn:
    _option_orders_disabled()


def place_options_combo_order(*args: Any, **kwargs: Any) -> NoReturn:
    _option_orders_disabled()


def place_options_combo_limit_order(*args: Any, **kwargs: Any) -> NoReturn:
    _option_orders_disabled()


def place_options_combo_market_order(*args: Any, **kwargs: Any) -> NoReturn:
    _option_orders_disabled()
