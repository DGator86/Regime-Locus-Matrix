"""Historical **stock** bars from Interactive Brokers (TWS / IB Gateway).

Use this for **equity OHLCV** when Massive range/trades are not entitled on your plan.
**Options** remain on Massive (``massive_option_chain_from_client``, snapshots).

Requires TWS or IB Gateway running with **API** enabled and matching host/port
(see ``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_CLIENT_ID`` in ``.env``).

Install: ``pip install 'regime-locus-matrix[ibkr]'`` (pulls PyPI ``ibapi``).

Docs: `TWS API <https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/>`_.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Type

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_ibkr_bundle: tuple[Type[Any], Any] | None = None


def load_ibkr_socket_config() -> tuple[str, int, int]:
    """Read ``IBKR_HOST``, ``IBKR_PORT``, ``IBKR_CLIENT_ID`` from environment (after ``load_dotenv``)."""
    load_dotenv()
    host = (os.environ.get("IBKR_HOST") or "127.0.0.1").strip()
    port = int(os.environ.get("IBKR_PORT") or "7497")
    client_id = int(os.environ.get("IBKR_CLIENT_ID") or "1")
    return host, port, client_id


def _bar_field(bar: Any, name: str, default: Any = None) -> Any:
    if isinstance(bar, dict):
        return bar.get(name, default)
    return getattr(bar, name, default)


def _bar_vwap(bar: Any) -> float:
    """Historical ``BarData`` uses ``average`` for bar VWAP; real-time bars use ``wap``."""
    for key in ("wap", "average"):
        v = _bar_field(bar, key, None)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x < 0:
            continue
        if x == 0.0 and key == "wap":
            continue
        return x
    return float(np.nan)


def ibkr_bars_to_dataframe(bars: list[Any]) -> pd.DataFrame:
    """Convert IB ``BarData`` objects (or dicts with same keys) to the Massive-aligned bar schema."""
    records: list[dict[str, Any]] = []
    for bar in bars:
        date_str = _bar_field(bar, "date")
        if not date_str:
            continue
        ts = pd.to_datetime(str(date_str).strip(), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        ts_naive = ts.tz_convert(None) if ts.tzinfo else ts
        wap_f = _bar_vwap(bar)
        records.append(
            {
                "timestamp": ts_naive,
                "open": float(_bar_field(bar, "open", np.nan)),
                "high": float(_bar_field(bar, "high", np.nan)),
                "low": float(_bar_field(bar, "low", np.nan)),
                "close": float(_bar_field(bar, "close", np.nan)),
                "volume": float(_bar_field(bar, "volume", 0) or 0),
                "vwap": wap_f,
            }
        )

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

    out = pd.DataFrame.from_records(records)
    return out.sort_values("timestamp").reset_index(drop=True)


def _us_stock_contract(symbol: str, exchange: str, currency: str, contract_cls: Any) -> Any:
    c = contract_cls()
    c.symbol = symbol.upper()
    c.secType = "STK"
    c.exchange = exchange
    c.currency = currency
    return c


# Non-fatal / informational codes we ignore for req completion
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


def _get_ibkr_bundle() -> tuple[Type[Any], Any]:
    """Lazy-load ``ibapi`` (app class + Contract). Raises a clear ImportError if missing."""
    global _ibkr_bundle
    if _ibkr_bundle is not None:
        return _ibkr_bundle

    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as e:
        raise ImportError(
            "IBKR historical bars require the official ibapi package. Install with: "
            "pip install 'regime-locus-matrix[ibkr]'"
        ) from e

    class _HistoricalBarsApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.bars: list[Any] = []
            self._ready = threading.Event()
            self._done = threading.Event()
            self._handshake_failed = threading.Event()
            self._req_id = 1
            self._last_error: tuple[int, str] | None = None
            # Last N IB messages (reqId, code, text) for debugging timeouts
            self._error_lines: list[tuple[int, int, str]] = []

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            self._ready.set()

        def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
            self.bars.append(bar)

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
            self._done.set()

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

            # System / connection channel (reqId -1): fail handshake fast when TWS rejects the client
            if reqId == -1:
                self._last_error = (errorCode, errorString)
                self._handshake_failed.set()
                return

            if reqId == self._req_id and errorCode not in _IB_INFO_ERROR_CODES:
                self._last_error = (errorCode, errorString)
                self._done.set()

    _ibkr_bundle = (_HistoricalBarsApp, Contract)
    return _ibkr_bundle


def fetch_historical_stock_bars(
    symbol: str,
    *,
    duration: str = "5 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    end_datetime: str = "",
    exchange: str = "SMART",
    currency: str = "USD",
    host: str | None = None,
    port: int | None = None,
    client_id: int | None = None,
    timeout_sec: float = 120.0,
) -> pd.DataFrame:
    """Request historical bars for a US stock and return a Massive-aligned DataFrame.

    ``duration`` / ``bar_size`` use IB strings (e.g. ``\"5 D\"``, ``\"1 day\"``, ``\"1 hour\"``).
    ``end_datetime`` empty means \"now\" per IB; otherwise e.g. ``\"20240105 16:00:00 US/Eastern\"``.

    TWS default ports: paper ``7497``, live ``7496``; Gateway: paper ``4002``, live ``4001``.
    """
    app_cls, Contract = _get_ibkr_bundle()
    h, p, cid = load_ibkr_socket_config()
    h = host if host is not None else h
    p = port if port is not None else p
    import random

    cid = (client_id if client_id is not None else cid) + random.randint(1000, 9999)

    app = app_cls()
    thread: threading.Thread | None = None
    try:
        # ``EClient.run()`` only loops while ``isConnected()`` or the queue is non-empty.
        # If ``run()`` starts before ``connect()``, the loop exits immediately and no API
        # messages (including ``nextValidId``) are ever processed. Connect first, then ``run()`` in a thread.
        app.connect(h, p, cid)
        thread = threading.Thread(target=app.run, daemon=False, name="ibapi-run")
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
                "Confirm TWS is logged in (same session as API port), click 'Accept' on any API popup, "
                f"and trusted IPs allow 127.0.0.1.{tail}"
            )

        contract = _us_stock_contract(symbol, exchange=exchange, currency=currency, contract_cls=Contract)
        app.reqHistoricalData(
            app._req_id,
            contract,
            end_datetime,
            duration,
            bar_size,
            what_to_show,
            use_rth,
            1,
            False,
            [],
        )

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if app._done.wait(timeout=0.25):
                break
        else:
            raise TimeoutError(f"IBKR: historical data timeout for {symbol!r} after {timeout_sec}s")

        if app._last_error is not None:
            code, msg = app._last_error
            raise RuntimeError(f"IBKR error {code}: {msg}")

        return ibkr_bars_to_dataframe(app.bars)
    finally:
        try:
            app.disconnect()
        except Exception:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=15.0)
        time.sleep(0.05)
