"""Place **multi-leg option** combo orders via Interactive Brokers (``ibapi``).

Uses a **BAG** contract built from per-leg ``OPT`` contracts resolved with
``reqContractDetails``. Paper TWS typically uses port **7497**; live **7496**.

**Safety:** Live ports are blocked unless ``acknowledge_live=True``. Default
``transmit=False`` parks the order in TWS for review unless you pass
``transmit=True``.

Requires TWS or IB Gateway with **API** enabled and options trading permissions.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Literal, Type

import pandas as pd
from dotenv import load_dotenv

IBKRLegAction = Literal["BUY", "SELL"]

# TWS default ports
IBKR_LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})
# 4004: gnzsnz ib-gateway-docker host-network + socat (paper API published on host)
IBKR_PAPER_PORTS: frozenset[int] = frozenset({7497, 4002, 4004})

_INFO_CODES = frozenset({2104, 2106, 2107, 2108, 2158, 2174})


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


def legs_from_ibkr_combo_spec(
    spec: dict[str, Any],
) -> list[tuple[IBKROptionLegSpec, IBKRLegAction]]:
    """Build IB leg tuples from ``ibkr_combo_spec`` JSON (``underlying``, ``legs[]``)."""
    und = str(spec.get("underlying", "")).upper().strip()
    if not und:
        raise ValueError("ibkr_combo_spec missing underlying")
    raw_legs = spec.get("legs")
    if not isinstance(raw_legs, list) or not raw_legs:
        raise ValueError("ibkr_combo_spec.legs must be a non-empty list")
    out: list[tuple[IBKROptionLegSpec, IBKRLegAction]] = []
    for leg in raw_legs:
        if not isinstance(leg, dict):
            raise ValueError("each leg must be an object")
        sp = IBKROptionLegSpec(
            underlying=und,
            expiry_yyyymmdd=expiry_iso_to_ib(str(leg["expiry"])),
            strike=float(leg["strike"]),
            right=option_type_to_ib_right(str(leg["option_type"])),
        )
        act = roee_side_to_ib_action(str(leg["side"]))
        out.append((sp, act))
    return out


def reverse_legs_for_close(
    legs: list[tuple[IBKROptionLegSpec, IBKRLegAction]],
) -> list[tuple[IBKROptionLegSpec, IBKRLegAction]]:
    """Flip BUY/SELL per leg (close an existing combo)."""
    rev: list[tuple[IBKROptionLegSpec, IBKRLegAction]] = []
    for spec, act in legs:
        new_act: IBKRLegAction = "SELL" if act == "BUY" else "BUY"
        rev.append((spec, new_act))
    return rev


def expiry_iso_to_ib(expiry: str) -> str:
    """``YYYY-MM-DD`` or ``YYYYMMDD`` → ``YYYYMMDD`` for IB ``lastTradeDateOrContractMonth``."""
    s = str(expiry).strip().replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    return pd.Timestamp(expiry).strftime("%Y%m%d")


def roee_side_to_ib_action(side: str) -> IBKRLegAction:
    s = str(side).lower().strip()
    if s == "long":
        return "BUY"
    if s == "short":
        return "SELL"
    raise ValueError(f"leg side must be 'long' or 'short', got {side!r}")


def option_type_to_ib_right(option_type: str) -> str:
    t = str(option_type).lower().strip()
    if t == "call":
        return "C"
    if t == "put":
        return "P"
    raise ValueError(f"option_type must be call or put, got {option_type!r}")


@dataclass(frozen=True)
class IBKROptionLegSpec:
    """Single option leg before ``conId`` resolution."""

    underlying: str
    expiry_yyyymmdd: str
    strike: float
    right: str  # "C" or "P"
    exchange: str = "SMART"
    currency: str = "USD"
    multiplier: str = "100"


def _get_bundle() -> tuple[Type[Any], Type[Any], Any, Any, Any]:
    try:
        from ibapi.client import EClient
        from ibapi.contract import ComboLeg, Contract
        from ibapi.order import Order
        from ibapi.wrapper import EWrapper
    except ImportError as e:
        raise ImportError("IBKR orders require ibapi. Install: pip install 'regime-locus-matrix[ibkr]'") from e
    return EWrapper, EClient, Contract, ComboLeg, Order


def _bare_option_contract(Contract: Any, spec: IBKROptionLegSpec) -> Any:
    c = Contract()
    c.symbol = spec.underlying.upper().strip()
    c.secType = "OPT"
    c.lastTradeDateOrContractMonth = spec.expiry_yyyymmdd
    c.strike = float(spec.strike)
    c.right = spec.right
    c.multiplier = spec.multiplier
    c.exchange = spec.exchange
    c.currency = spec.currency
    return c


def _pick_resolved_contract(Contract: Any, details_list: list[Any]) -> Any:
    if not details_list:
        raise RuntimeError("No contract details returned from IBKR.")
    contracts = [d.contract for d in details_list]
    smart = [x for x in contracts if getattr(x, "exchange", "") == "SMART" and getattr(x, "secType", "") == "OPT"]
    chosen = smart[0] if smart else contracts[0]
    if not getattr(chosen, "conId", 0):
        raise RuntimeError(f"Resolved contract has no conId: {chosen}")
    return chosen


class _ComboOrderApp:
    """EWrapper + EClient merged at runtime (same pattern as ``ibkr_stocks``)."""

    def __init__(self) -> None:
        EWrapper, EClient, _, _, _ = _get_bundle()

        class App(EWrapper, EClient):
            def __init__(self) -> None:
                EClient.__init__(self, self)
                self._ready = threading.Event()
                self._handshake_failed = threading.Event()
                self._next_order_id: int | None = None
                self._last_error: tuple[int, str] | None = None
                self._error_lines: list[tuple[int, int, str]] = []

                self._cd_req_id: int = -1
                self._cd_bucket: list[Any] = []
                self._cd_done = threading.Event()
                self._cd_seq = 8000

                self._order_status: dict[int, list[str]] = {}

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                self._next_order_id = int(orderId)
                self._ready.set()

            def error(  # noqa: N802
                self,
                reqId: int,
                errorCode: int,
                errorString: str,
                advancedOrderRejectJson: str = "",
            ) -> None:
                self._error_lines.append((reqId, errorCode, errorString))
                if len(self._error_lines) > 80:
                    self._error_lines = self._error_lines[-80:]
                if errorCode in _INFO_CODES:
                    return
                if reqId == -1:
                    self._last_error = (errorCode, errorString)
                    self._handshake_failed.set()
                    return
                if reqId == self._cd_req_id:
                    self._cd_done.set()
                    if errorCode not in _INFO_CODES:
                        self._last_error = (errorCode, errorString)

            def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
                if reqId == self._cd_req_id:
                    self._cd_bucket.append(contractDetails)

            def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
                if reqId == self._cd_req_id:
                    self._cd_done.set()

            def orderStatus(  # noqa: N802
                self,
                orderId: int,
                status: str,
                filled: float,
                remaining: float,
                avgFillPrice: float,
                permId: int,
                parentId: int,
                lastFillPrice: float,
                clientId: int,
                whyHeld: str,
                mktCapPrice: float,
            ) -> None:
                self._order_status.setdefault(orderId, []).append(str(status))

        self._app_cls = App
        self.app: Any = App()


def _wait_handshake(app: Any, host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + min(timeout, 30.0)
    while time.monotonic() < deadline:
        if app._ready.is_set():
            return
        if app._handshake_failed.is_set() and app._last_error is not None:
            code, msg = app._last_error
            raise RuntimeError(f"IBKR handshake failed ({code}): {msg}")
        time.sleep(0.05)
    raise RuntimeError(
        f"IBKR: no nextValidId from {host}:{port} within timeout. "
        "Confirm TWS/Gateway is logged in and API is enabled."
    )


def _connect_run(app: Any, host: str, port: int, client_id: int, timeout: float) -> threading.Thread:
    app.connect(host, port, client_id)
    thread = threading.Thread(target=app.run, daemon=False, name="ibkr-order-run")
    thread.start()
    _wait_handshake(app, host, port, timeout)
    return thread


def _disconnect(app: Any, thread: threading.Thread | None) -> None:
    try:
        app.disconnect()
    except Exception:
        pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=20.0)
    time.sleep(0.05)


@contextmanager
def ibkr_order_connection(
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    *,
    timeout_sec: float = 60.0,
) -> Generator[Any, None, None]:
    h, p, cid = load_ibkr_order_socket_config()
    h = host if host is not None else h
    p = port if port is not None else p
    cid = client_id if client_id is not None else cid

    shell = _ComboOrderApp()
    app = shell.app
    thread: threading.Thread | None = None
    try:
        thread = _connect_run(app, h, p, cid, timeout_sec)
        yield app
    finally:
        _disconnect(app, thread)


def resolve_option_contract(
    spec: IBKROptionLegSpec,
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 45.0,
) -> Any:
    """One-shot: connect, resolve, disconnect. Returns ``Contract`` with ``conId``."""
    _, _, Contract, _, _ = _get_bundle()
    with ibkr_order_connection(host=host, port=port, client_id=client_id, timeout_sec=timeout_sec) as app:
        return _resolve_option_on_app(app, Contract, spec, timeout_sec)


def _resolve_option_on_app(app: Any, Contract: Any, spec: IBKROptionLegSpec, timeout_sec: float) -> Any:
    bare = _bare_option_contract(Contract, spec)
    app._cd_seq += 1
    req_id = app._cd_seq
    app._cd_req_id = req_id
    app._cd_bucket = []
    app._cd_done.clear()
    app._last_error = None

    app.reqContractDetails(req_id, bare)
    if not app._cd_done.wait(timeout=timeout_sec):
        raise TimeoutError(f"IBKR contractDetails timeout for {spec}")
    if app._last_error is not None and not app._cd_bucket:
        code, msg = app._last_error
        raise RuntimeError(f"IBKR contract resolution failed ({code}): {msg}")
    resolved = _pick_resolved_contract(Contract, app._cd_bucket)
    time.sleep(0.25)  # pacing between detail requests when caller resolves multiple legs
    return resolved


def place_options_combo_order(
    legs: list[tuple[IBKROptionLegSpec, IBKRLegAction]],
    *,
    quantity: int,
    transmit: bool = False,
    acknowledge_live: bool = False,
    combo_order_action: IBKRLegAction = "BUY",
    order_type: Literal["LMT", "MKT"] = "LMT",
    limit_price: float | None = None,
    account: str | None = None,
    tif: str = "DAY",
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 90.0,
) -> tuple[int, list[str]]:
    """
    Resolve each leg, build a **BAG**, and ``placeOrder`` (limit or market).

    ``combo_order_action``: **BUY** for typical net-debit opens; **SELL** to close that package.
    """
    if quantity < 1:
        raise ValueError("quantity must be >= 1")
    if not legs:
        raise ValueError("At least one leg is required")
    ot = str(order_type).upper()
    if ot == "LMT" and limit_price is None:
        raise ValueError("limit_price is required for LMT orders")
    h, p, cid = load_ibkr_order_socket_config()
    h = host if host is not None else h
    p = port if port is not None else p
    cid = client_id if client_id is not None else cid

    assert_paper_or_live_acknowledged(p, acknowledge_live=acknowledge_live)

    _, _, Contract, ComboLeg, Order = _get_bundle()
    underlying0 = legs[0][0].underlying.upper()

    with ibkr_order_connection(host=h, port=p, client_id=cid, timeout_sec=timeout_sec) as app:
        combo_contracts: list[tuple[Any, IBKRLegAction]] = []
        for spec, action in legs:
            if spec.underlying.upper() != underlying0:
                raise ValueError("All legs must share the same underlying for a single BAG combo.")
            rc = _resolve_option_on_app(app, Contract, spec, timeout_sec)
            combo_contracts.append((rc, action))

        bag = Contract()
        bag.symbol = underlying0
        bag.secType = "BAG"
        bag.currency = legs[0][0].currency
        bag.exchange = "SMART"

        combo_legs: list[Any] = []
        for rc, action in combo_contracts:
            leg = ComboLeg()
            leg.conId = int(rc.conId)
            leg.ratio = 1
            leg.action = action
            leg.exchange = "SMART"
            combo_legs.append(leg)
        bag.comboLegs = combo_legs

        if app._next_order_id is None:
            raise RuntimeError("IBKR did not provide nextValidId")
        order_id = int(app._next_order_id)
        app._next_order_id = order_id + 1

        order = Order()
        order.action = combo_order_action
        order.totalQuantity = float(quantity)
        order.orderType = ot
        if ot == "LMT":
            order.lmtPrice = float(limit_price)  # type: ignore[arg-type]
        order.transmit = bool(transmit)
        order.tif = str(tif)
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        if account:
            order.account = str(account)

        app._order_status.pop(order_id, None)
        app.placeOrder(order_id, bag, order)

        trail = _wait_order_terminal(app, order_id, transmit=transmit, timeout_sec=timeout_sec)
        return order_id, trail


def place_options_combo_limit_order(
    legs: list[tuple[IBKROptionLegSpec, IBKRLegAction]],
    *,
    quantity: int,
    limit_price: float,
    transmit: bool = False,
    acknowledge_live: bool = False,
    combo_order_action: IBKRLegAction = "BUY",
    account: str | None = None,
    tif: str = "DAY",
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 90.0,
) -> tuple[int, list[str]]:
    """Limit-only wrapper around :func:`place_options_combo_order`."""
    return place_options_combo_order(
        legs,
        quantity=quantity,
        transmit=transmit,
        acknowledge_live=acknowledge_live,
        combo_order_action=combo_order_action,
        order_type="LMT",
        limit_price=limit_price,
        account=account,
        tif=tif,
        host=host,
        port=port,
        client_id=client_id,
        timeout_sec=timeout_sec,
    )


def place_options_combo_market_order(
    legs: list[tuple[IBKROptionLegSpec, IBKRLegAction]],
    *,
    quantity: int,
    transmit: bool = True,
    acknowledge_live: bool = False,
    combo_order_action: IBKRLegAction = "SELL",
    account: str | None = None,
    tif: str = "DAY",
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 90.0,
) -> tuple[int, list[str]]:
    """Market combo (typical use: **paper** close of a debit structure with parent **SELL**)."""
    return place_options_combo_order(
        legs,
        quantity=quantity,
        transmit=transmit,
        acknowledge_live=acknowledge_live,
        combo_order_action=combo_order_action,
        order_type="MKT",
        limit_price=None,
        account=account,
        tif=tif,
        host=host,
        port=port,
        client_id=client_id,
        timeout_sec=timeout_sec,
    )


def _wait_order_terminal(app: Any, order_id: int, *, transmit: bool, timeout_sec: float) -> list[str]:
    deadline = time.monotonic() + timeout_sec
    trail: list[str] = []
    while time.monotonic() < deadline:
        trail = list(app._order_status.get(order_id, []))
        if trail:
            last = trail[-1]
            if last in ("Filled", "Cancelled", "ApiCancelled"):
                return trail
            if transmit and last in ("Submitted", "PreSubmitted"):
                return trail
            if not transmit and last in ("PreSubmitted", "Submitted", "Inactive", "PendingSubmit"):
                return trail
        # Return early if IBKR sent an error for this specific order (e.g., order rejected).
        # The error() callback only stores reqId != -1 errors in _error_lines.
        if any(r == order_id for r, _, _ in app._error_lines):
            errs = [(c, m) for r, c, m in app._error_lines if r == order_id]
            raise RuntimeError(f"IBKR order {order_id} error: {errs[-1]}")
        time.sleep(0.05)
    return trail
