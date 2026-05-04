"""Regime-directed equity paper trading via IBKR.

Reads ``universe_trade_plans.json`` (output of ``run_universe_options_pipeline.py``),
extracts the regime direction for every active plan, and places simple stock
BUY / SELL orders on the IBKR paper account.

This runs alongside the options book for independent execution verification:
- No options permissions required — plain equity orders work at any account level.
- Bull regime  → BUY shares
- Bear regime  → SELL (short) shares
- Range / other → skip

Positions are tracked in ``equity_positions_state.json``.  On each run the
script also evaluates open equity positions and closes those whose underlying
plan is no longer active, whose regime has flipped, or whose stop/target is hit.

Usage
-----
    python scripts/ibkr_equity_paper_trade.py
    python scripts/ibkr_equity_paper_trade.py --dry-run          # no real orders
    python scripts/ibkr_equity_paper_trade.py --position-usd 5000
    python scripts/ibkr_equity_paper_trade.py --stop-pct 3 --target-pct 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Type

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Optional ibapi dependency — required for live IBKR connectivity, not needed for --dry-run.
try:
    from ibapi.client import EClient as _EClient
    from ibapi.wrapper import EWrapper as _EWrapper
    from ibapi.contract import Contract as _IbkrContract
    from ibapi.order import Order as _IbkrOrder
    _IBAPI_OK = True
except ImportError:
    _EClient = _EWrapper = _IbkrContract = _IbkrOrder = None  # type: ignore[assignment,misc]
    _IBAPI_OK = False

from rlm.utils.compute_threads import apply_compute_thread_env  # noqa: E402
apply_compute_thread_env()

from rlm.data.ibkr_snapshot import fetch_ibkr_account_snapshot

PLANS_PATH = ROOT / "data" / "processed" / "universe_trade_plans.json"
EQUITY_STATE_PATH = ROOT / "data" / "processed" / "equity_positions_state.json"
EQUITY_LOG_PATH = ROOT / "data" / "processed" / "equity_trade_log.csv"

IBKR_LIVE_PORTS: frozenset[int] = frozenset({7496, 4001})
IBKR_PAPER_PORTS: frozenset[int] = frozenset({7497, 4002, 4004})

# IBKR error codes that are advisory notices, not hard order rejections.
# These are silently swallowed by the error handler so they never enter
# _error_lines and never trigger false-positive rejection raises.
#   2104-2174  market-data / connectivity info
#   10349      "Order TIF was set to DAY based on order preset"
#   10314      "Order modified to comply with …"
#   10197      "No market data during competing session"
_IBKR_ADVISORY_CODES: frozenset[int] = frozenset({
    2104, 2106, 2107, 2108, 2158, 2174,
    10197, 10314, 10349,
})

_LOG_COLUMNS = [
    "timestamp_utc", "plan_id", "symbol", "strategy", "action",
    "quantity", "current_mark", "entry_debit", "order_id",
    "unrealized_pnl", "unrealized_pnl_pct", "signal", "closed", "note",
]


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class EquityPosition:
    plan_id: str
    symbol: str
    direction: str          # "bull" | "bear"
    side: str               # "long" | "short"
    quantity: int
    entry_price: float
    entry_ts: str           # ISO UTC
    ibkr_order_id: int | None = None
    close_order_id: int | None = None
    status: str = "open"    # "open" | "closed"
    exit_price: float | None = None
    exit_ts: str | None = None
    exit_reason: str | None = None


def _load_state(path: Path) -> dict[str, EquityPosition]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, EquityPosition] = {}
    for pid, d in raw.items():
        try:
            out[pid] = EquityPosition(**d)
        except TypeError:
            pass
    return out


def _save_state(positions: dict[str, EquityPosition], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {pid: asdict(pos) for pid, pos in positions.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV trade log
# ---------------------------------------------------------------------------

def _append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# IBKR connectivity (stock orders)
# ---------------------------------------------------------------------------

def _load_equity_socket_config() -> tuple[str, int, int]:
    host = (os.environ.get("IBKR_HOST") or "127.0.0.1").strip()
    port = int(os.environ.get("IBKR_PORT") or "7497")
    cid = int(os.environ.get("IBKR_EQUITY_CLIENT_ID") or "10")
    return host, port, cid


def _get_ibapi_bundle() -> tuple[Type[Any], Any]:
    """Return (EClient, EWrapper); raise SystemExit if ibapi is not installed."""
    if not _IBAPI_OK:
        raise SystemExit("ibapi not installed. Run: pip install ibapi")
    return _EClient, _EWrapper


class _EquityApp:
    """Minimal IBKR EWrapper/EClient combo for stock orders."""

    def __init__(self) -> None:
        EClient, EWrapper = _get_ibapi_bundle()

        class _App(EWrapper, EClient):  # type: ignore[misc]
            def __init__(inner_self) -> None:
                EWrapper.__init__(inner_self)
                EClient.__init__(inner_self, inner_self)

        self._app = _App()
        self._app._order_status: dict[int, list[str]] = {}
        self._app._error_lines: list[tuple[int, int, str]] = []
        self._app._next_order_id: int | None = None
        self._app._ticker_prices: dict[int, float] = {}
        self._app._ticker_events: dict[int, threading.Event] = {}
        # Separate counter for market-data reqIds so they never collide with order IDs.
        # Order IDs start from whatever IBKR assigns (typically small integers) and
        # count upward.  Market-data reqIds start at 10_000 and count downward,
        # keeping them well away from the order-ID sequence in both directions.
        self._mkt_req_counter: int = 10_000

        original_error = self._app.error.__func__ if hasattr(self._app.error, "__func__") else None

        def _error(reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
            if errorCode in _IBKR_ADVISORY_CODES:
                return  # purely informational — do not record or print
            print(f"  [ibkr-err] reqId={reqId} code={errorCode} {errorString}", flush=True)
            if reqId != -1:
                self._app._error_lines.append((reqId, errorCode, errorString))

        def _order_status(orderId: int, status: str, filled: float, remaining: float,
                          avgFillPrice: float, permId: int, parentId: int, lastFillPrice: float,
                          clientId: int, whyHeld: str, mktCapPrice: float = 0.0) -> None:
            self._app._order_status.setdefault(orderId, []).append(status)

        def _next_valid_id(orderId: int) -> None:
            self._app._next_order_id = orderId

        def _tick_price(reqId: int, tickType: int, price: float, attrib: Any) -> None:
            if tickType in (1, 2, 4, 68):  # bid, ask, last, midpoint
                self._app._ticker_prices[reqId] = price
                if reqId in self._app._ticker_events:
                    self._app._ticker_events[reqId].set()

        self._app.error = _error  # type: ignore[method-assign]
        self._app.orderStatus = _order_status  # type: ignore[method-assign]
        self._app.nextValidId = _next_valid_id  # type: ignore[method-assign]
        self._app.tickPrice = _tick_price  # type: ignore[method-assign]

    @property
    def app(self) -> Any:
        return self._app

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, clientId=client_id)
        t = threading.Thread(target=self._app.run, daemon=True)
        t.start()
        deadline = time.monotonic() + 15.0
        while self._app._next_order_id is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._app._next_order_id is None:
            raise RuntimeError("IBKR handshake timed out — is TWS/Gateway running?")
        print(f"  [equity-ibkr] connected client_id={self._app.clientId}, "
              f"next_order_id={self._app._next_order_id}", flush=True)

    def disconnect(self) -> None:
        try:
            self._app.disconnect()
        except Exception:
            pass

    def next_order_id(self) -> int:
        oid = self._app._next_order_id
        if oid is None:
            raise RuntimeError("Not connected to IBKR")
        self._app._next_order_id = oid + 1
        return oid

    def _next_mkt_req_id(self) -> int:
        """Return a market-data reqId that is separate from the order-ID sequence.

        Counts down from 10_000 so market-data reqIds move away from order IDs
        (which start low and increment upward from IBKR's nextValidId).
        """
        req_id = self._mkt_req_counter
        self._mkt_req_counter -= 1
        return req_id

    def get_last_price(self, symbol: str, timeout_sec: float = 8.0) -> float | None:
        """Request live last price via reqMktData tick snapshot.

        Uses a dedicated reqId counter (10_000+) so market-data requests never
        advance the order-ID sequence.  Returns None gracefully if the account
        lacks a real-time data subscription (IBKR error 10089).
        """
        if not _IBAPI_OK or _IbkrContract is None:
            return None
        req_id = self._next_mkt_req_id()
        contract = _IbkrContract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        ev = threading.Event()
        self._app._ticker_events[req_id] = ev

        # Clear any stale error for this reqId before subscribing.
        self._app._error_lines = [
            (r, c, m) for r, c, m in self._app._error_lines if r != req_id
        ]

        self._app.reqMktData(req_id, contract, "", True, False, [])

        # Poll in short increments so we can bail out early on error 10089
        # (no market-data subscription) rather than blocking the full timeout.
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if ev.wait(timeout=0.25):
                break
            # Check for a subscription-denial error on this reqId
            if any(r == req_id and c in (10089, 354) for r, c, _ in self._app._error_lines):
                break

        try:
            self._app.cancelMktData(req_id)
        except Exception:
            pass

        price = self._app._ticker_prices.get(req_id)
        return price

    def place_stock_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        limit_price: float | None = None,
        transmit: bool = True,
    ) -> int:
        if not _IBAPI_OK:
            raise SystemExit("ibapi not installed. Run: pip install ibapi")

        contract = _IbkrContract()  # type: ignore[misc]
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = _IbkrOrder()  # type: ignore[misc]
        order.action = action  # "BUY" or "SELL"
        order.totalQuantity = quantity
        if limit_price is not None:
            order.orderType = "LMT"
            order.lmtPrice = round(limit_price, 2)
        else:
            order.orderType = "MKT"
        order.transmit = transmit
        # ibapi sets eTradeOnly=True and firmQuoteOnly=True by default; IBKR
        # paper accounts reject both with error 10268. Clear them explicitly.
        order.eTradeOnly = False
        order.firmQuoteOnly = False

        oid = self.next_order_id()
        self._app.placeOrder(oid, contract, order)
        return oid

    def wait_for_order(self, order_id: int, timeout_sec: float = 30.0) -> list[str]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            trail = list(self._app._order_status.get(order_id, []))
            if trail:
                last = trail[-1]
                if last == "Rejected":
                    errs = [(c, m) for r, c, m in self._app._error_lines if r == order_id]
                    raise RuntimeError(
                        f"IBKR order {order_id} rejected: {errs[-1] if errs else 'Rejected'}"
                    )
                if last in ("Filled", "Cancelled", "ApiCancelled", "Submitted", "PreSubmitted"):
                    return trail
            # Raise only on hard (non-advisory) error codes — advisory codes are
            # already filtered at the _error callback, but guard here too.
            hard_errs = [
                (c, m) for r, c, m in self._app._error_lines
                if r == order_id and c not in _IBKR_ADVISORY_CODES
            ]
            if hard_errs:
                raise RuntimeError(f"IBKR order {order_id} rejected: {hard_errs[-1]}")
            time.sleep(0.1)
        return list(self._app._order_status.get(order_id, []))


@contextmanager
def ibkr_equity_connection() -> Generator[_EquityApp, None, None]:
    host, port, cid = _load_equity_socket_config()
    if port in IBKR_LIVE_PORTS:
        raise ValueError(
            f"Refusing automated equity orders on live port {port}. "
            "Set IBKR_PORT to a paper port (7497 / 4002)."
        )
    app = _EquityApp()
    app.connect(host, port, cid)
    try:
        yield app
    finally:
        app.disconnect()


# ---------------------------------------------------------------------------
# Plan reading helpers
# ---------------------------------------------------------------------------

def _load_plans(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    plans: list[dict] = []
    seen_ids: set[str] = set()
    for row in (raw.get("active_ranked") or []) + (raw.get("results") or []):
        if not (isinstance(row, dict) and row.get("status") == "active"):
            continue
        pid = str(row.get("plan_id") or "")
        if pid and pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        plans.append(row)
    return plans


def _mark_equity_opened(plan_id: str, plans_path: Path) -> None:
    """Stamp equity_opened=true on the plan in the JSON file."""
    if not plans_path.is_file():
        return
    try:
        payload = json.loads(plans_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    changed = False
    for section in ("active_ranked", "results"):
        for row in payload.get(section) or []:
            if isinstance(row, dict) and row.get("plan_id") == plan_id:
                row["equity_opened"] = True
                changed = True
    if changed:
        plans_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _quantity_for_symbol(
    price: float,
    position_usd: float,
    risk_usd: float | None = None,
    stop_pct: float | None = None,
    account_nlv: float | None = None,
    max_account_pct: float | None = None,
    confidence: float | None = None,
) -> int:
    if price <= 0:
        return 0

    # Option 1: Account-percentage scaling (Notional = NLV * MaxPct * Confidence)
    if account_nlv is not None and max_account_pct is not None:
        conf = confidence if confidence is not None else 1.0
        target_notional = account_nlv * (max_account_pct / 100.0) * conf
        qty = int(target_notional // price)
        return max(1, qty)

    # Option 2: Risk-based sizing (Risk $ = Fixed Risk * Confidence)
    if risk_usd is not None and risk_usd > 0 and stop_pct is not None and stop_pct > 0:
        # Quantity = Risk $ / (Price * Stop %)
        conf = confidence if confidence is not None else 1.0
        scaled_risk = risk_usd * conf
        qty = int(scaled_risk / (price * (stop_pct / 100.0)))
        return max(1, qty)

    # Option 3: Fixed notional
    return max(1, int(position_usd // price))


def open_equity_positions(
    *,
    plans: list[dict],
    positions: dict[str, EquityPosition],
    position_usd: float,
    risk_usd: float | None = None,
    stop_pct: float | None = None,
    account_nlv: float | None = None,
    max_account_pct: float | None = None,
    dry_run: bool,
    app: _EquityApp | None,
    plans_path: Path,
    log_path: Path,
) -> None:
    open_symbols = {pos.symbol.upper() for pos in positions.values() if pos.status == "open"}

    for plan in plans:
        sym = str(plan.get("symbol", "")).upper()
        plan_id = str(plan.get("plan_id", ""))
        if not sym or not plan_id:
            continue
        if plan.get("equity_opened"):
            continue  # already entered
        if sym in open_symbols:
            print(f"  [equity] {sym}: already have open position — skip", flush=True)
            continue

        direction = str(plan.get("regime_direction", "")).lower()
        if direction not in ("bull", "bear"):
            print(f"  [equity] {sym}: direction={direction!r} → skip (not bull/bear)", flush=True)
            continue

        action = "BUY" if direction == "bull" else "SELL"
        side = "long" if direction == "bull" else "short"

        # Determine entry price — pipeline stores it under plan["pipeline"]["close"]
        pipeline_data = plan.get("pipeline") or {}
        entry_price = float(
            pipeline_data.get("close")
            or plan.get("close")
            or plan.get("current_price")
            or 0.0
        )
        if entry_price <= 0 and app is not None:
            print(f"  [equity] {sym}: fetching live price …", flush=True)
            lp = app.get_last_price(sym)
            if lp and lp > 0:
                entry_price = lp
        if entry_price <= 0:
            print(f"  [equity] {sym}: cannot determine price — skip", flush=True)
            continue

        # Extract confidence from plan metadata
        decision_data = plan.get("decision") or {}
        meta = decision_data.get("metadata") or {}
        confidence = float(
            meta.get("regime_confidence")
            or meta.get("confidence")
            or decision_data.get("size_fraction")
            or 1.0
        )

        qty = _quantity_for_symbol(
            entry_price,
            position_usd,
            risk_usd=risk_usd,
            stop_pct=stop_pct,
            account_nlv=account_nlv,
            max_account_pct=max_account_pct,
            confidence=confidence,
        )
        if qty == 0:
            print(f"  [equity] {sym}: qty=0 at price={entry_price:.2f} — skip", flush=True)
            continue

        print(
            f"  [equity] {sym}: {action} {qty} shares @ ~${entry_price:.2f} "
            f"(${qty * entry_price:,.0f} notional) [dry={dry_run}]",
            flush=True,
        )

        order_id: int | None = None
        if not dry_run and app is not None:
            try:
                # Use a slight slippage limit for shorts, market for longs
                lim = round(entry_price * (0.998 if direction == "bear" else 1.002), 2)
                order_id = app.place_stock_order(sym, action, qty, limit_price=lim)
                trail = app.wait_for_order(order_id)
                print(f"    order_id={order_id} trail={trail}", flush=True)
            except Exception as exc:
                print(f"    [equity] {sym}: order error — {exc}", flush=True)
                continue

        pos = EquityPosition(
            plan_id=plan_id,
            symbol=sym,
            direction=direction,
            side=side,
            quantity=qty,
            entry_price=entry_price,
            entry_ts=datetime.now(tz=timezone.utc).isoformat(),
            ibkr_order_id=order_id,
            status="open" if (dry_run or order_id is not None) else "pending",
        )
        positions[plan_id] = pos
        open_symbols.add(sym)

        if not dry_run and order_id is not None:
            _mark_equity_opened(plan_id, plans_path)

        _append_log(log_path, {
            "timestamp_utc": pos.entry_ts,
            "plan_id": plan_id,
            "symbol": sym,
            "strategy": direction,
            "action": action,
            "quantity": qty,
            "current_mark": entry_price,
            "entry_debit": entry_price,
            "order_id": order_id or "",
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "signal": "open",
            "closed": "0",
            "note": "dry_run" if dry_run else "placed",
        })


def evaluate_equity_positions(
    *,
    positions: dict[str, EquityPosition],
    active_plan_ids: set[str],
    stop_pct: float,
    target_pct: float,
    dry_run: bool,
    app: _EquityApp | None,
    log_path: Path,
) -> None:
    for plan_id, pos in list(positions.items()):
        if pos.status != "open":
            continue

        # Get current price
        current_price: float | None = None
        if app is not None:
            current_price = app.get_last_price(pos.symbol)
        if current_price is None or current_price <= 0:
            current_price = pos.entry_price  # fallback — no change

        # P&L calculation
        if pos.side == "long":
            pnl = (current_price - pos.entry_price) * pos.quantity
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100.0
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100.0

        # Determine exit signal
        exit_reason: str | None = None
        if plan_id not in active_plan_ids:
            exit_reason = "plan_no_longer_active"
        elif pnl_pct <= -stop_pct:
            exit_reason = f"stop_loss_{stop_pct}pct"
        elif pnl_pct >= target_pct:
            exit_reason = f"take_profit_{target_pct}pct"

        signal = exit_reason or "hold"
        print(
            f"  [equity-monitor] {pos.symbol}: side={pos.side} "
            f"entry={pos.entry_price:.2f} current={current_price:.2f} "
            f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%) signal={signal}",
            flush=True,
        )

        _append_log(log_path, {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "plan_id": plan_id,
            "symbol": pos.symbol,
            "strategy": pos.direction,
            "action": "hold",
            "quantity": pos.quantity,
            "current_mark": current_price,
            "entry_debit": pos.entry_price,
            "order_id": pos.ibkr_order_id or "",
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 4),
            "signal": signal,
            "closed": "0",
            "note": "",
        })

        if exit_reason is None:
            continue

        # Close position
        close_action = "SELL" if pos.side == "long" else "BUY"
        print(f"  [equity] CLOSING {pos.symbol}: {close_action} {pos.quantity} shares — {exit_reason} [dry={dry_run}]", flush=True)

        close_order_id: int | None = None
        if not dry_run and app is not None:
            try:
                close_order_id = app.place_stock_order(pos.symbol, close_action, pos.quantity)
                trail = app.wait_for_order(close_order_id)
                print(f"    close order_id={close_order_id} trail={trail}", flush=True)
            except Exception as exc:
                print(f"    [equity] {pos.symbol}: close order error — {exc}", flush=True)

        pos.status = "closed"
        pos.exit_price = current_price
        pos.exit_ts = datetime.now(tz=timezone.utc).isoformat()
        pos.exit_reason = exit_reason
        pos.close_order_id = close_order_id

        _append_log(log_path, {
            "timestamp_utc": pos.exit_ts,
            "plan_id": plan_id,
            "symbol": pos.symbol,
            "strategy": pos.direction,
            "action": close_action,
            "quantity": pos.quantity,
            "current_mark": current_price,
            "entry_debit": pos.entry_price,
            "order_id": close_order_id or "",
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 4),
            "signal": "closed",
            "closed": "1",
            "note": exit_reason,
        })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR equity paper trade from regime plans")
    parser.add_argument("--plans", default=str(PLANS_PATH), help="Path to universe_trade_plans.json")
    parser.add_argument("--state", default=str(EQUITY_STATE_PATH), help="Path to equity positions state JSON")
    parser.add_argument("--log", default=str(EQUITY_LOG_PATH), help="Path to equity trade log CSV")
    parser.add_argument("--position-usd", type=float, default=10_000.0,
                        help="Target notional USD per position (default: $10,000; ignored if --risk-usd is set)")
    parser.add_argument("--risk-usd", type=float, default=None,
                        help="Dollar amount to risk per trade (e.g. 500). If set, overrides --position-usd.")
    parser.add_argument("--use-account-scale", action="store_true",
                        help="Scale position size based on account balance (NLV).")
    parser.add_argument("--max-account-pct", type=float, default=10.0,
                        help="Max percentage of account balance per position (default: 10.0)")
    parser.add_argument("--stop-pct", type=float, default=5.0,
                        help="Hard stop loss %% below entry (default: 5)")
    parser.add_argument("--target-pct", type=float, default=10.0,
                        help="Take-profit %% above entry (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Paper-mode without IBKR orders (log only)")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip opening new positions; only evaluate existing ones")
    args = parser.parse_args()

    plans_path = Path(args.plans)
    state_path = Path(args.state)
    log_path = Path(args.log)

    print(f"\n{'='*60}", flush=True)
    print(f"  IBKR Equity Paper Trade  |  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    print(f"  plans       : {plans_path}", flush=True)
    print(f"  position_usd: ${args.position_usd:,.0f}{' (ignored)' if args.risk_usd or args.use_account_scale else ''}", flush=True)
    if args.risk_usd:
        print(f"  risk_usd    : ${args.risk_usd:,.0f}", flush=True)
    if args.use_account_scale:
        print(f"  account_scale: max {args.max_account_pct}% of NLV scaled by confidence", flush=True)
    print(f"  stop / target: -{args.stop_pct}% / +{args.target_pct}%", flush=True)
    print(f"  dry_run     : {args.dry_run}", flush=True)
    print(f"{'='*60}\n", flush=True)

    plans = _load_plans(plans_path)
    positions = _load_state(state_path)
    active_plan_ids = {p["plan_id"] for p in plans if p.get("plan_id")}

    account_nlv: float | None = None
    if args.use_account_scale:
        try:
            print("[equity] Fetching account balance for scaling …", flush=True)
            snap = fetch_ibkr_account_snapshot(timeout_sec=30.0)
            for row in snap.account_summary:
                if row.tag == "NetLiquidation":
                    try:
                        account_nlv = float(row.value)
                        print(f"  [equity] Account NLV: ${account_nlv:,.2f}", flush=True)
                        break
                    except (ValueError, TypeError):
                        pass
            if account_nlv is None:
                print("  [equity] [warn] Could not find NetLiquidation in account summary. Falling back to default sizing.", flush=True)
        except Exception as e:
            print(f"  [equity] [error] Could not fetch account snapshot: {e}. Falling back to default sizing.", flush=True)

    print(f"[equity] Loaded {len(plans)} active plans, {len(positions)} tracked positions", flush=True)

    if args.dry_run:
        # No IBKR connection needed — work in dry-run mode
        if not args.monitor_only:
            open_equity_positions(
                plans=plans, positions=positions, position_usd=args.position_usd,
                risk_usd=args.risk_usd, stop_pct=args.stop_pct,
                account_nlv=account_nlv, max_account_pct=args.max_account_pct,
                dry_run=True, app=None, plans_path=plans_path, log_path=log_path,
            )
        evaluate_equity_positions(
            positions=positions, active_plan_ids=active_plan_ids,
            stop_pct=args.stop_pct, target_pct=args.target_pct,
            dry_run=True, app=None, log_path=log_path,
        )
        _save_state(positions, state_path)
        print("\n[equity] dry-run complete.", flush=True)
        return

    # Live IBKR connection
    with ibkr_equity_connection() as app:
        if not args.monitor_only:
            open_equity_positions(
                plans=plans, positions=positions, position_usd=args.position_usd,
                risk_usd=args.risk_usd, stop_pct=args.stop_pct,
                account_nlv=account_nlv, max_account_pct=args.max_account_pct,
                dry_run=False, app=app, plans_path=plans_path, log_path=log_path,
            )
        evaluate_equity_positions(
            positions=positions, active_plan_ids=active_plan_ids,
            stop_pct=args.stop_pct, target_pct=args.target_pct,
            dry_run=False, app=app, log_path=log_path,
        )

    _save_state(positions, state_path)
    print("\n[equity] done.", flush=True)


if __name__ == "__main__":
    main()
