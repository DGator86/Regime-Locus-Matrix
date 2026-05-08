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
script evaluates open equity positions against the **fresh** universe row for
each ``plan_id``: ROEE regime flip (:func:`~rlm.roee.exits.should_exit_for_regime_flip`),
optional min top-1 **transition-matrix** next-step probability
(``RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB``), optional second-layer
**label-aligned** mass on that same one-step vector (``RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS``;
see ``pipeline.regime_transition.next_label_aligned_*_mass`` in the universe row),
percentage stop/target, and plan-universe absence (``RLM_EQUITY_PLAN_MISSING_GRACE_SEC``).

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
from dataclasses import dataclass, asdict
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
from rlm.regimes.forecast_regime_snapshot import (
    plan_regime_key,
    position_directional_transition_mass,
    regime_direction_equity,
    regime_transition_best_prob,
)
from rlm.roee.exits import should_exit_for_regime_flip
from rlm.universe.active_plans import iter_active_trade_plan_rows  # noqa: E402

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
    plan_missing_since_utc: str | None = None
    entry_regime_key: str = ""


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


def _parse_iso_utc(ts: str) -> datetime:
    s = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _plan_missing_grace_sec(cli_value: float | None) -> float:
    if cli_value is not None:
        return max(0.0, float(cli_value))
    raw = (os.environ.get("RLM_EQUITY_PLAN_MISSING_GRACE_SEC") or "").strip()
    if raw:
        return max(0.0, float(raw))
    return 900.0


def _plans_by_plan_id(plans: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in plans:
        pid = str(p.get("plan_id") or "")
        if pid:
            out[pid] = p
    return out


def _merge_universe_rows_for_open_positions(
    plan_by_id: dict[str, dict],
    plans_path: Path,
    positions: dict[str, EquityPosition],
) -> None:
    """Fill plan snapshots for open legs whose plan_id is not in the active set (e.g. trimmed row)."""
    missing = {
        pid
        for pid, pos in positions.items()
        if pos.status == "open" and pid not in plan_by_id
    }
    if not missing or not plans_path.is_file():
        return
    try:
        raw = json.loads(plans_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for row in (raw.get("active_ranked") or []) + (raw.get("results") or []):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plan_id") or "")
        if pid in missing:
            plan_by_id[pid] = row


def _min_most_likely_next_prob(cli_value: float | None) -> float | None:
    """Optional threshold on calibrated/top-1 next-step prob (unset = disable)."""
    if cli_value is not None:
        v = float(cli_value)
        return v if v > 0.0 else None
    raw = (os.environ.get("RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB") or "").strip()
    if not raw:
        return None
    v = float(raw)
    return v if v > 0.0 else None


def _min_next_label_aligned_mass(cli_value: float | None) -> float | None:
    """Optional min mass on Σ P(next_state)×label_alignment for traded direction."""
    if cli_value is not None:
        v = float(cli_value)
        return v if v > 0.0 else None
    raw = (os.environ.get("RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS") or "").strip()
    if not raw:
        return None
    v = float(raw)
    return v if v > 0.0 else None


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
    if not isinstance(raw, dict):
        return []
    return iter_active_trade_plan_rows(raw)


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

        rk_entry = plan_regime_key(plan)
        direction = str(plan.get("regime_direction") or regime_direction_equity(rk_entry) or "").lower()
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
            entry_regime_key=rk_entry,
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
    plan_by_id: dict[str, dict],
    stop_pct: float,
    target_pct: float,
    grace_sec: float,
    min_most_likely_next_prob: float | None,
    min_next_label_aligned_mass: float | None,
    dry_run: bool,
    app: _EquityApp | None,
    log_path: Path,
    utc_now: datetime | None = None,
) -> None:
    now = utc_now if utc_now is not None else datetime.now(tz=timezone.utc)
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

        if plan_id in active_plan_ids:
            pos.plan_missing_since_utc = None

        cur_plan_raw = plan_by_id.get(plan_id)
        cur_plan: dict | None = cur_plan_raw if isinstance(cur_plan_raw, dict) else None
        pipe: dict[str, Any] = {}
        if isinstance(cur_plan, dict):
            p_raw = cur_plan.get("pipeline")
            if isinstance(p_raw, dict):
                pipe = p_raw
        rt = pipe.get("regime_transition")
        trans_snap: dict[str, Any] | None = rt if isinstance(rt, dict) else None
        p_next = regime_transition_best_prob(trans_snap)
        cur_rk = plan_regime_key(cur_plan) if cur_plan else ""
        aligned_mass = position_directional_transition_mass(trans_snap, pos.direction)

        # Price risk first; active-plan regime + transition; then universe absence grace.
        exit_reason: str | None = None
        if pnl_pct <= -stop_pct:
            exit_reason = f"stop_loss_{stop_pct}pct"
        elif pnl_pct >= target_pct:
            exit_reason = f"take_profit_{target_pct}pct"

        ek = str(pos.entry_regime_key or "").strip()
        if exit_reason is None and cur_plan and ek and cur_rk and should_exit_for_regime_flip(ek, cur_rk):
            exit_reason = "regime_flip"

        if (
            exit_reason is None
            and cur_plan
            and min_most_likely_next_prob is not None
            and p_next is not None
            and p_next < float(min_most_likely_next_prob)
        ):
            exit_reason = "weak_transition_top1_prob"

        if (
            exit_reason is None
            and cur_plan
            and min_next_label_aligned_mass is not None
            and aligned_mass is not None
            and aligned_mass < float(min_next_label_aligned_mass)
        ):
            exit_reason = "weak_transition_label_mass"

        if exit_reason is None and plan_id not in active_plan_ids:
            if grace_sec <= 0.0:
                exit_reason = "plan_no_longer_active"
            else:
                if pos.plan_missing_since_utc is None:
                    pos.plan_missing_since_utc = now.isoformat()
                absent_for = (now - _parse_iso_utc(pos.plan_missing_since_utc)).total_seconds()
                if absent_for >= grace_sec:
                    exit_reason = "plan_no_longer_active"

        signal = exit_reason or "hold"
        grace_note = ""
        if (
            exit_reason is None
            and plan_id not in active_plan_ids
            and grace_sec > 0.0
            and pos.plan_missing_since_utc is not None
        ):
            absent_for = (now - _parse_iso_utc(pos.plan_missing_since_utc)).total_seconds()
            grace_note = f" (plan absent {absent_for:.0f}s / {grace_sec:.0f}s grace)"
        trans_note = f" transition_top1_p={p_next:.4f}" if p_next is not None else ""
        rk_note = f" cur_regime={cur_rk[:40]}" if cur_rk else ""
        mass_note = ""
        if aligned_mass is not None:
            mass_note = f" label_aligned_mass({pos.direction})={aligned_mass:.3f}"
        print(
            f"  [equity-monitor] {pos.symbol}: side={pos.side} "
            f"entry={pos.entry_price:.2f} current={current_price:.2f} "
            f"pnl=${pnl:+.2f} ({pnl_pct:+.2f}%) signal={signal}{grace_note}{trans_note}{mass_note}{rk_note}",
            flush=True,
        )

        _append_log(log_path, {
            "timestamp_utc": now.isoformat(),
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
        pos.exit_ts = now.isoformat()
        pos.exit_reason = exit_reason
        pos.close_order_id = close_order_id
        pos.plan_missing_since_utc = None

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
    parser.add_argument(
        "--plan-missing-grace-sec",
        type=float,
        default=None,
        help=(
            "Seconds a plan may be absent from the active universe before equity auto-close "
            "(default: env RLM_EQUITY_PLAN_MISSING_GRACE_SEC or 900). Use 0 for immediate close."
        ),
    )
    parser.add_argument(
        "--min-most-likely-next-prob",
        type=float,
        default=None,
        help=(
            "Exit stock leg if calibrated/top-1 next-step regime transition probability falls "
            "below this threshold (unset = disabled; overrides env "
            "RLM_EQUITY_MIN_MOST_LIKELY_NEXT_PROB)."
        ),
    )
    parser.add_argument(
        "--min-next-label-aligned-mass",
        type=float,
        default=None,
        help=(
            "Exit when Σ P(next state)×label-weight for the traded direction falls below this "
            "(0–1; unset = disabled; overrides RLM_EQUITY_MIN_NEXT_LABEL_ALIGNED_MASS). "
            "Uses full transition row + HMM/Markov state labels from the universe pipeline."
        ),
    )
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
    grace_sec = _plan_missing_grace_sec(args.plan_missing_grace_sec)
    min_np = _min_most_likely_next_prob(args.min_most_likely_next_prob)
    min_lm = _min_next_label_aligned_mass(args.min_next_label_aligned_mass)
    print(f"  stop / target : -{args.stop_pct}% / +{args.target_pct}%", flush=True)
    print(f"  plan missing grace (universe): {grace_sec:.0f}s", flush=True)
    print(
        f"  min transition top-1 p: {'%.4f (active)' % min_np if min_np else 'disabled'}",
        flush=True,
    )
    print(
        f"  min label-aligned mass: {'%.4f (active)' % min_lm if min_lm else 'disabled'}",
        flush=True,
    )
    print(f"  dry_run     : {args.dry_run}", flush=True)
    print(f"{'='*60}\n", flush=True)

    plans = _load_plans(plans_path)
    positions = _load_state(state_path)
    active_plan_ids = {p["plan_id"] for p in plans if p.get("plan_id")}
    plan_by_id = _plans_by_plan_id(plans)
    _merge_universe_rows_for_open_positions(plan_by_id, plans_path, positions)

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
            positions=positions,
            active_plan_ids=active_plan_ids,
            plan_by_id=plan_by_id,
            stop_pct=args.stop_pct,
            target_pct=args.target_pct,
            grace_sec=grace_sec,
            min_most_likely_next_prob=min_np,
            min_next_label_aligned_mass=min_lm,
            dry_run=True,
            app=None,
            log_path=log_path,
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
            positions=positions,
            active_plan_ids=active_plan_ids,
            plan_by_id=plan_by_id,
            stop_pct=args.stop_pct,
            target_pct=args.target_pct,
            grace_sec=grace_sec,
            min_most_likely_next_prob=min_np,
            min_next_label_aligned_mass=min_lm,
            dry_run=False,
            app=app,
            log_path=log_path,
        )

    _save_state(positions, state_path)
    print("\n[equity] done.", flush=True)


if __name__ == "__main__":
    main()
