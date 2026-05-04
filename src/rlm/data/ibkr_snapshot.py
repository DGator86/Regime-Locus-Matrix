"""One-shot **account summary** and **positions** from TWS / IB Gateway (read-only).

Uses ``reqAccountSummary`` and ``reqPositions`` over the standard IBKR socket API.
Requires ``pip install 'regime-locus-matrix[ibkr]'`` and a running TWS/Gateway session.

Socket env (via ``load_ibkr_socket_config``): ``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_CLIENT_ID``.
Dashboard / tooling should use ``IBKR_DASHBOARD_CLIENT_ID`` when set; otherwise
``IBKR_CLIENT_ID + 20`` to reduce clashes with historical bars or order clients.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Type

from rlm.data.ibkr_stocks import load_ibkr_socket_config

# Tags accepted by ``reqAccountSummary`` (comma-separated).
ACCOUNT_SUMMARY_TAGS = (
    "NetLiquidation,TotalCashValue,BuyingPower,GrossPositionValue,AvailableFunds,"
    "UnrealizedPnL,RealizedPnL,ExcessLiquidity,FullInitMarginReq"
)

_IB_INFO_ERROR_CODES = frozenset(
    {
        2104,
        2106,
        2107,
        2108,
        2158,
        2174,
    }
)


def _format_ib_error_tail(lines: list[tuple[int, int, str]], limit: int = 12) -> str:
    if not lines:
        return ""
    tail = lines[-limit:]
    body = "\n".join(f"    reqId={r} code={c} {s}" for r, c, s in tail)
    return f"\n  Recent IB API messages:\n{body}"


def load_ibkr_dashboard_socket_config() -> tuple[str, int, int]:
    """Host/port from ``load_ibkr_socket_config``; client id from ``IBKR_DASHBOARD_CLIENT_ID`` or ``base+20``."""
    host, port, base_cid = load_ibkr_socket_config()
    raw = os.environ.get("IBKR_DASHBOARD_CLIENT_ID")
    if raw is not None and str(raw).strip() != "":
        return host, port, int(raw)
    return host, port, int(base_cid) + 20


@dataclass(frozen=True)
class IbkrPositionRow:
    account: str
    symbol: str
    local_symbol: str
    sec_type: str
    currency: str
    exchange: str
    con_id: int
    position: float
    avg_cost: float


@dataclass(frozen=True)
class IbkrAccountSummaryRow:
    account: str
    tag: str
    value: str
    currency: str


@dataclass(frozen=True)
class IbkrSnapshot:
    positions: tuple[IbkrPositionRow, ...]
    account_summary: tuple[IbkrAccountSummaryRow, ...]
    host: str
    port: int
    client_id: int


_ibkr_types: tuple[Type[Any], Any] | None = None


def _get_ibkr_types() -> tuple[Type[Any], Any]:
    global _ibkr_types
    if _ibkr_types is not None:
        return _ibkr_types
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ImportError as e:
        raise ImportError(
            "IBKR snapshot requires the official ibapi package. Install with: "
            "pip install 'regime-locus-matrix[ibkr]'"
        ) from e

    class _SnapshotApp(EWrapper, EClient):
        SUMMARY_REQ_ID = 9001

        def __init__(self) -> None:
            EClient.__init__(self, self)
            self._ready = threading.Event()
            self._handshake_failed = threading.Event()
            self._last_error: tuple[int, str] | None = None
            self._error_lines: list[tuple[int, int, str]] = []
            self._positions: list[IbkrPositionRow] = []
            self._pos_done = threading.Event()
            self._summary: list[IbkrAccountSummaryRow] = []
            self._sum_done = threading.Event()

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            self._ready.set()

        def position(  # noqa: N802
            self,
            account: str,
            contract: Any,
            position: float,
            avgCost: float,
        ) -> None:
            sym = str(getattr(contract, "symbol", "") or "")
            loc = str(getattr(contract, "localSymbol", "") or "")
            st = str(getattr(contract, "secType", "") or "")
            ccy = str(getattr(contract, "currency", "") or "")
            exch = str(getattr(contract, "exchange", "") or "")
            try:
                cid = int(getattr(contract, "conId", 0) or 0)
            except (TypeError, ValueError):
                cid = 0
            self._positions.append(
                IbkrPositionRow(
                    account=str(account or ""),
                    symbol=sym,
                    local_symbol=loc,
                    sec_type=st,
                    currency=ccy,
                    exchange=exch,
                    con_id=cid,
                    position=float(position),
                    avg_cost=float(avgCost),
                )
            )

        def positionEnd(self) -> None:  # noqa: N802
            self._pos_done.set()

        def accountSummary(  # noqa: N802
            self,
            reqId: int,
            account: str,
            tag: str,
            value: str,
            currency: str,
        ) -> None:
            if int(reqId) == self.SUMMARY_REQ_ID:
                self._summary.append(
                    IbkrAccountSummaryRow(
                        account=str(account or ""),
                        tag=str(tag or ""),
                        value=str(value or ""),
                        currency=str(currency or ""),
                    )
                )

        def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802
            if int(reqId) == self.SUMMARY_REQ_ID:
                self._sum_done.set()

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

            if errorCode in _IB_INFO_ERROR_CODES:
                return

            if int(reqId) == -1:
                self._last_error = (errorCode, errorString)
                self._handshake_failed.set()
                return

            if int(reqId) == self.SUMMARY_REQ_ID and errorCode not in _IB_INFO_ERROR_CODES:
                self._last_error = (errorCode, errorString)
                self._sum_done.set()

    _ibkr_types = (_SnapshotApp, None)
    return _ibkr_types


def fetch_ibkr_account_snapshot(
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 25.0,
) -> IbkrSnapshot:
    """Connect, pull account summary + positions, disconnect. Raises on timeout or IB error."""
    app_cls, _ = _get_ibkr_types()
    h, p, cid = load_ibkr_dashboard_socket_config()
    h = host if host is not None else h
    p = port if port is not None else p
    import random
    cid = (client_id if client_id is not None else cid) + random.randint(1000, 9999)

    app = app_cls()
    thread: threading.Thread | None = None
    try:
        app.connect(h, p, cid)
        thread = threading.Thread(target=app.run, daemon=False, name="ibapi-snapshot-run")
        thread.start()

        handshake_deadline = time.monotonic() + min(timeout_sec, 30.0)
        while time.monotonic() < handshake_deadline:
            if app._ready.is_set():
                break
            if app._handshake_failed.is_set() and app._last_error is not None:
                code, msg = app._last_error
                tail = _format_ib_error_tail(app._error_lines)
                raise RuntimeError(f"IBKR handshake failed ({code}): {msg}.{tail}")
            time.sleep(0.05)
        else:
            tail = _format_ib_error_tail(app._error_lines)
            raise RuntimeError(
                f"IBKR: no nextValidId from {h}:{p} within timeout. "
                "Confirm TWS/Gateway is logged in and API is enabled."
                f"{tail}"
            )

        app._pos_done.clear()
        app._sum_done.clear()
        app._positions.clear()
        app._summary.clear()
        app._last_error = None

        app.reqAccountSummary(app.SUMMARY_REQ_ID, "All", ACCOUNT_SUMMARY_TAGS)
        app.reqPositions()

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if app._pos_done.is_set() and app._sum_done.is_set():
                break
            if app._handshake_failed.is_set() and app._last_error is not None:
                code, msg = app._last_error
                tail = _format_ib_error_tail(app._error_lines)
                raise RuntimeError(f"IBKR error during snapshot ({code}): {msg}.{tail}")
            time.sleep(0.05)
        else:
            tail = _format_ib_error_tail(app._error_lines)
            raise TimeoutError(
                f"IBKR: snapshot timeout after {timeout_sec}s (positions_done={app._pos_done.is_set()} "
                f"summary_done={app._sum_done.is_set()}).{tail}"
            )

        if app._last_error is not None:
            code, msg = app._last_error
            raise RuntimeError(f"IBKR error {code}: {msg}")

        try:
            app.cancelAccountSummary(app.SUMMARY_REQ_ID)
        except Exception:
            pass

        return IbkrSnapshot(
            positions=tuple(app._positions),
            account_summary=tuple(app._summary),
            host=h,
            port=p,
            client_id=cid,
        )
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
        time.sleep(0.05)
