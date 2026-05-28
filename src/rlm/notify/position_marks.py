"""Live mark refresh before Telegram / positions reports."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rlm.challenge.live_marks import refresh_challenge_at_root
from rlm.data.occ_symbol import format_occ_compact_symbol
from rlm.market.live_quotes import fetch_equity_quotes, live_marks_enabled
from rlm.roee.chain_match import (
    estimate_mark_value_from_matched_legs,
    leg_contract_key,
    refresh_matched_leg_mids,
)


@dataclass
class LiveOptionOverlay:
    current_mark: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    underlying_last: float | None = None
    leg_lines: list[str] = field(default_factory=list)
    asof_utc: str | None = None


@dataclass
class LiveEquityOverlay:
    mark_price: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    asof_utc: str | None = None


@dataclass
class LiveMarksBundle:
    asof_utc: str | None = None
    options: dict[str, LiveOptionOverlay] = field(default_factory=dict)
    equities: dict[str, LiveEquityOverlay] = field(default_factory=dict)
    challenge_asof: str | None = None


def refresh_all_position_marks(root: Path) -> LiveMarksBundle:
    """Fetch live prices and persist challenge marks; return display overlays."""
    bundle = LiveMarksBundle()
    if not live_marks_enabled():
        return bundle

    asof = datetime.now(tz=timezone.utc).isoformat()
    bundle.asof_utc = asof
    bundle.challenge_asof = refresh_challenge_at_root(root)

    from rlm.notify.telegram_rlm import default_paths  # noqa: PLC0415

    p = default_paths(root)
    bundle.options = _refresh_large_options_overlays(root, p)
    bundle.equities = _refresh_equity_overlays(p)
    return bundle


def _refresh_large_options_overlays(root: Path, p: dict[str, Path]) -> dict[str, LiveOptionOverlay]:
    from rlm.notify.telegram_rlm import (  # noqa: PLC0415
        _latest_rows_per_plan_csv,
        _load_trade_plan_snapshots,
        _plan_with_trade_log_legs,
        _read_plans,
        _resolved_options_plan,
    )

    overlays: dict[str, LiveOptionOverlay] = {}
    latest = _latest_rows_per_plan_csv(p["trade_log"])
    open_rows = [(pid, row) for pid, row in latest.items() if (row.get("closed") or "0").strip() != "1"]
    if not open_rows:
        return overlays

    plans_data = _read_plans(p["plans"]) or {}
    snapshots = _load_trade_plan_snapshots(p["trade_plan_snapshots"])
    symbols = sorted({str(row.get("symbol") or "").upper() for _, row in open_rows if row.get("symbol")})

    chains: dict[str, pd.DataFrame] = _load_option_chains(symbols)
    spot = fetch_equity_quotes(symbols)

    for pid, row in open_rows:
        sym = str(row.get("symbol") or "").upper()
        plan = _plan_with_trade_log_legs(
            _resolved_options_plan(plans_data, pid, snapshots),
            row,
        )
        legs = _ensure_leg_contract_symbols(sym, [x for x in (plan.get("matched_legs") or []) if isinstance(x, dict)])
        if not legs:
            continue

        chain = chains.get(sym, pd.DataFrame())
        updated = refresh_matched_leg_mids(chain, legs) if not chain.empty else None
        entry_debit = float(plan.get("entry_debit_dollars") or row.get("entry_debit") or 0.0)
        und = spot.get(sym)

        ov = LiveOptionOverlay(asof_utc=und.asof_utc if und else None)
        if und is not None:
            ov.underlying_last = und.price

        if updated is not None:
            v = float(estimate_mark_value_from_matched_legs(updated))
            pnl = v - entry_debit
            pnl_pct = (pnl / abs(entry_debit) * 100.0) if abs(entry_debit) > 1e-6 else None
            ov.current_mark = v
            ov.unrealized_pnl = pnl
            ov.unrealized_pnl_pct = pnl_pct
            for lg in updated:
                side = str(lg.get("side") or "").upper() or "—"
                sk = lg.get("strike", "?")
                ot = str(lg.get("option_type") or "?")[:1].upper()
                mid = float(lg.get("mid") or 0)
                ov.leg_lines.append(f"      {side} ${sk} {ot} mid ${mid:.2f}/sh")
        overlays[pid] = ov

    return overlays


def _refresh_equity_overlays(p: dict[str, Path]) -> dict[str, LiveEquityOverlay]:
    import json

    from rlm.notify.telegram_rlm import _latest_equity_open_rows_by_plan  # noqa: PLC0415

    overlays: dict[str, LiveEquityOverlay] = {}
    eq_path = p.get("equity_state")
    if eq_path is None or not eq_path.is_file():
        return overlays
    try:
        eq = json.loads(eq_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return overlays

    open_items = [
        (pid, d)
        for pid, d in eq.items()
        if isinstance(d, dict) and str(d.get("status") or "") == "open"
    ]
    if not open_items:
        return overlays

    symbols = sorted({str(d.get("symbol") or "").upper() for _, d in open_items})
    quotes = fetch_equity_quotes(symbols)
    eq_log = _latest_equity_open_rows_by_plan(p["equity_trade_log"])

    for pid, d in open_items:
        sym = str(d.get("symbol") or "").upper()
        q = quotes.get(sym)
        if q is None:
            continue
        try:
            entry = float(d.get("entry_price") or 0)
            qty = float(d.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        if entry <= 0 or qty == 0:
            continue
        side = str(d.get("side") or "long").lower()
        sign = 1.0 if side == "long" else -1.0
        pnl = (q.price - entry) * qty * sign
        pnl_pct = (q.price / entry - 1.0) * 100.0 * sign if entry > 0 else None
        overlays[pid] = LiveEquityOverlay(
            mark_price=q.price,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl_pct,
            asof_utc=q.asof_utc,
        )
    return overlays


def _load_option_chains(symbols: list[str]) -> dict[str, pd.DataFrame]:
    chains: dict[str, pd.DataFrame] = {}
    massive_key = (os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or "").strip()
    if massive_key:
        try:
            from rlm.data.massive import MassiveClient
            from rlm.data.massive_option_chain import massive_option_chains_from_client

            client = MassiveClient(api_key=massive_key)
            batch = massive_option_chains_from_client(client, symbols, cache_ttl_s=30.0)
            chains.update(batch.chains)
        except Exception:  # noqa: BLE001
            pass

    for sym in symbols:
        if sym in chains and not chains[sym].empty:
            continue
        frame = _yfinance_chain_for_symbol(sym)
        if not frame.empty:
            chains[sym] = frame
    return chains


def _yfinance_chain_for_symbol(symbol: str) -> pd.DataFrame:
    """Build a minimal chain frame with ``contract_symbol`` for leg matching."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        return pd.DataFrame()

    sym = str(symbol).upper()
    records: list[dict[str, Any]] = []
    try:
        t = yf.Ticker(sym)
        for exp in list(t.options or [])[:8]:
            try:
                exp_d = date.fromisoformat(str(exp)[:10])
            except ValueError:
                continue
            ch = t.option_chain(exp)
            for side_name, frame in (("call", ch.calls), ("put", ch.puts)):
                if frame is None or frame.empty:
                    continue
                for _, row in frame.iterrows():
                    strike = float(row.get("strike", 0))
                    bid = _f(row.get("bid"))
                    ask = _f(row.get("ask"))
                    if bid is None and ask is None:
                        continue
                    bid = bid if bid is not None else 0.01
                    ask = ask if ask is not None else bid
                    mid = (bid + ask) / 2.0
                    occ = format_occ_compact_symbol(sym, exp_d, option_type=side_name, strike=strike)
                    records.append(
                        {
                            "contract_symbol": f"O:{occ}",
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "strike": strike,
                            "expiry": exp_d.isoformat(),
                            "option_type": side_name,
                        }
                    )
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    return pd.DataFrame(records) if records else pd.DataFrame()


def _ensure_leg_contract_symbols(underlying: str, legs: list[dict]) -> list[dict]:
    """Fill ``symbol`` / ``contract_symbol`` when plans only store strike + expiry."""
    out: list[dict] = []
    for lg in legs:
        d = dict(lg)
        if leg_contract_key(d):
            out.append(d)
            continue
        try:
            exp = date.fromisoformat(str(d.get("expiry"))[:10])
            strike = float(d["strike"])
            ot = str(d.get("option_type") or "call")
            occ = format_occ_compact_symbol(underlying, exp, option_type=ot, strike=strike)
            d["contract_symbol"] = f"O:{occ}"
            d["symbol"] = d["contract_symbol"]
        except (KeyError, ValueError, TypeError):
            pass
        out.append(d)
    return out


def _f(val: Any) -> float | None:
    try:
        if val is None:
            return None
        x = float(val)
        return None if x != x else x
    except (TypeError, ValueError):
        return None
