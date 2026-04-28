#!/usr/bin/env python3
"""
Live P&L summary from the trade log written by monitor_active_trade_plans.py.

Usage::

    python scripts/show_pnl.py                        # summary of all trades
    python scripts/show_pnl.py --closed               # only closed (exited) trades
    python scripts/show_pnl.py --open                 # only open (active) trades
    python scripts/show_pnl.py --log data/processed/trade_log.csv

Columns shown
-------------
symbol   strategy   status   entry_debit  current_mark  unrealized_pnl  pnl_pct  signal  dte
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_per_plan(rows: list[dict]) -> list[dict]:
    """Keep only the most recent row for each plan_id."""
    seen: dict[str, dict] = {}
    for row in rows:
        seen[row["plan_id"]] = row
    return list(seen.values())


def _is_equity_log(rows: list[dict]) -> bool:
    """Return True when rows came from ibkr_equity_paper_trade (equity schema).

    Detected by the presence of ``action`` and ``quantity``, which are
    equity-specific columns absent from the options trade log.
    """
    if not rows:
        return False
    cols = set(rows[0].keys())
    return "action" in cols and "quantity" in cols


def _normalize_equity_rows(rows: list[dict]) -> list[dict]:
    """Project equity log rows to the options-display schema.

    The equity log now uses the same overlapping column names as the options
    log (``strategy``, ``entry_debit``, ``current_mark``, ``closed``), so
    most fields are direct pass-throughs.  Only ``quantity`` is remapped to
    the ``dte`` display slot (header is relabelled to QTY for equity views).
    """
    out = []
    for r in rows:
        out.append(
            {
                "plan_id": r.get("plan_id", ""),
                "symbol": r.get("symbol", ""),
                "strategy": r.get("strategy", ""),
                "closed": r.get("closed", "0"),
                "entry_debit": r.get("entry_debit", ""),
                "current_mark": r.get("current_mark", ""),
                "unrealized_pnl": r.get("unrealized_pnl", ""),
                "unrealized_pnl_pct": r.get("unrealized_pnl_pct", ""),
                "signal": r.get("signal", ""),
                "dte": r.get("quantity", ""),
            }
        )
    return out


def _fmt(val: str, width: int, align: str = ">") -> str:
    try:
        f = float(val)
        s = f"{f:,.2f}"
    except (ValueError, TypeError):
        s = str(val)
    if align == ">":
        return s.rjust(width)
    return s.ljust(width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("data/processed/trade_log.csv"),
        help="Options trade log (default: data/processed/trade_log.csv)",
    )
    ap.add_argument(
        "--equity",
        action="store_true",
        help="Show equity trade log (data/processed/equity_trade_log.csv)",
    )
    ap.add_argument("--equity-log", type=Path, default=None, help="Explicit path to equity trade log CSV")
    ap.add_argument("--closed", action="store_true", help="Show only closed trades")
    ap.add_argument("--open", dest="open_only", action="store_true", help="Show only open trades")
    ap.add_argument("--all-rows", action="store_true", help="Show every log row, not just latest per plan")
    args = ap.parse_args()

    if args.equity or args.equity_log:
        raw_path = args.equity_log or Path("data/processed/equity_trade_log.csv")
        log_path = ROOT / raw_path if not raw_path.is_absolute() else raw_path
    else:
        log_path = ROOT / args.log if not args.log.is_absolute() else args.log

    rows = _load_log(log_path)

    if not rows:
        print(f"No trade log found at {log_path}")
        print("The monitor writes to this file automatically once positions are being tracked.")
        return 0

    # Detect equity log by column presence and normalise to display schema.
    is_equity = _is_equity_log(rows)
    if is_equity:
        rows = _normalize_equity_rows(rows)

    if not args.all_rows:
        rows = _latest_per_plan(rows)

    if args.closed:
        rows = [r for r in rows if r.get("closed") == "1"]
    elif args.open_only:
        rows = [r for r in rows if r.get("closed") != "1"]

    if not rows:
        print("No matching rows.")
        return 0

    # Sort by symbol then plan_id
    rows.sort(key=lambda r: (r.get("symbol", ""), r.get("plan_id", "")))

    # Header — last column is DTE for options, QTY for equity.
    if is_equity:
        hdr = (
            f"{'SYMBOL':<7} {'DIRECTION':<24} {'STATUS':<7} "
            f"{'ENTRY_$':>12} {'CURR_$':>10} {'UNRLZD_PNL':>12} {'PNL_%':>7} "
            f"{'SIGNAL':<20} {'QTY':>6}"
        )
    else:
        hdr = (
            f"{'SYMBOL':<7} {'STRATEGY':<24} {'STATUS':<7} "
            f"{'ENTRY_DEBIT':>12} {'CURR_MARK':>10} {'UNRLZD_PNL':>12} {'PNL_%':>7} "
            f"{'SIGNAL':<20} {'DTE':>6}"
        )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    total_pnl = 0.0
    closed_pnl = 0.0
    n_open = n_closed = 0

    for r in rows:
        status = "CLOSED" if r.get("closed") == "1" else "open"
        pnl_raw = r.get("unrealized_pnl", "")
        try:
            pnl_f = float(pnl_raw)
            total_pnl += pnl_f
            if r.get("closed") == "1":
                closed_pnl += pnl_f
                n_closed += 1
            else:
                n_open += 1
        except (ValueError, TypeError):
            pnl_f = float("nan")

        pnl_pct = r.get("unrealized_pnl_pct", "")
        dte = r.get("dte", "")

        # Colour indicator (ASCII-safe)
        try:
            marker = " +" if pnl_f > 0 else (" -" if pnl_f < 0 else "  ")
        except Exception:
            marker = "  "

        print(
            f"{r.get('symbol',''):<7} "
            f"{r.get('strategy',''):<24} "
            f"{status:<7} "
            f"{_fmt(r.get('entry_debit',''), 12)} "
            f"{_fmt(r.get('current_mark',''), 10)} "
            f"{marker}{_fmt(pnl_raw, 10)} "
            f"{_fmt(pnl_pct, 6)}% "
            f"{r.get('signal',''):<20} "
            f"{_fmt(dte, 6)}"
        )

    print(sep)
    print(
        f"  Open: {n_open}   Closed: {n_closed}   "
        f"Total unrealized P&L: ${total_pnl:,.2f}   "
        f"Closed P&L: ${closed_pnl:,.2f}"
    )
    print(sep)
    if log_path.is_file():
        print(f"  Log: {log_path}  ({log_path.stat().st_size // 1024 + 1} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
