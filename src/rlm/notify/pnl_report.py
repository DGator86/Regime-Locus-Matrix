"""
EOD PnL Report — daily trading performance from `trade_log.csv`.

The log is a **per-poll append** of mark-to-model P&L (`unrealized_pnl` = mark
value minus entry debit). A high trade count and a low "win rate" on that
metric usually mean many positions are open and slightly under water, not
necessarily that 70 distinct trades lost after exit. This module separates
**open MTM** from **exits that fired** (TP/stop/ trail / expiry) so the report
is easier to read.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


def _now_utc() -> datetime:
    """Test hook: patch this instead of :func:`datetime.now`."""
    return datetime.now(timezone.utc)


def _row_session_date_utc(iso_ts: str) -> datetime | None:
    """Return UTC aware datetime from a trade_log `timestamp_utc` cell, or None."""
    raw = (iso_ts or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iter_trade_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _latest_per_plan_today(today_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Last row in file order per `plan_id` (monitor appends in time order)."""
    latest_by_plan: dict[str, dict[str, str]] = {}
    for r in today_rows:
        pid = str(r.get("plan_id", ""))
        if pid:
            latest_by_plan[pid] = r
    return latest_by_plan


def _pnl(r: dict[str, str]) -> float:
    try:
        return float(r.get("unrealized_pnl", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_closed(r: dict[str, str]) -> bool:
    return str(r.get("closed", "0")).strip() == "1"


def calculate_daily_pnl(root: Path) -> str:
    trade_log = root / "data" / "processed" / "trade_log.csv"
    if not trade_log.is_file():
        return "No trade log found for today."

    now_utc = _now_utc()
    session_date = now_utc.astimezone(_ET).date()
    pnl_sum = 0.0
    # Legacy headline stats: last mark per plan on this **session** (US/Eastern) day
    wins = 0
    losses = 0
    total_plans = 0

    try:
        all_rows: list[dict[str, str]] = list(_iter_trade_rows(trade_log))
        if not all_rows:
            return "Trade log is empty."

        today_rows: list[dict[str, str]] = []
        for r in all_rows:
            ts = _row_session_date_utc(r.get("timestamp_utc", ""))
            if ts is None:
                continue
            if ts.astimezone(_ET).date() == session_date:
                today_rows.append(r)

        if not today_rows:
            return f"No trading activity recorded for US/Eastern session {session_date}."

        latest_by_plan = _latest_per_plan_today(today_rows)

        open_mtm = 0.0
        open_n = 0
        open_wins = 0
        open_losses = 0

        closed_pnl = 0.0
        closed_n = 0
        c_wins = 0
        c_losses = 0
        sym_pnl: dict[str, float] = {}

        for _pid, r in latest_by_plan.items():
            p = _pnl(r)
            pnl_sum += p
            total_plans += 1
            sym = str(r.get("symbol", "") or "?")
            sym_pnl[sym] = sym_pnl.get(sym, 0.0) + p

            if p > 0:
                wins += 1
            elif p < 0:
                losses += 1

            if _is_closed(r):
                closed_pnl += p
                closed_n += 1
                if p > 0:
                    c_wins += 1
                elif p < 0:
                    c_losses += 1
            else:
                open_mtm += p
                open_n += 1
                if p > 0:
                    open_wins += 1
                elif p < 0:
                    open_losses += 1

        wr = (wins / total_plans * 100) if total_plans else 0.0
        closed_wr = (c_wins / closed_n * 100) if closed_n else 0.0

        worst_syms = sorted(sym_pnl.items(), key=lambda x: x[1])[:5]
        worst_line = ""
        if worst_syms:
            parts = [f"{s} {v:+.2f}" for s, v in worst_syms if v < 0]
            if parts:
                worst_line = "Worst MTM (symbol sum): " + ", ".join(parts) + "\n"

        mtm_note = (
            f"  (headline MTM win% {wr:.1f}% is mark vs entry, not round-trip; "
            f"see exit line for stop/TP quality.)"
        )
        if total_plans <= 30:
            mtm_note = ""

        return (
            f"<b>[EOD Report] {session_date} (ET)</b>\n"
            f"Total mark vs entry (last row per plan, session): <b>${pnl_sum:+.2f}</b>\n"
            f"Open: {open_n}  MTM ${open_mtm:+.2f}  (+{open_wins} / −{open_losses} vs entry)\n"
            f"Exits (closed=1): {closed_n}  at-exit P&L ${closed_pnl:+.2f}  "
            f"{c_wins}W / {c_losses}L  ({closed_wr:.1f}% of exits green){mtm_note}\n"
            f"unique plans: {total_plans}\n"
            f"{worst_line}"
        )
    except Exception as e:
        return f"Error generating PnL report: {e}"
