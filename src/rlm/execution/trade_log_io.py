"""Shared options paper ``trade_log.csv`` I/O (monitor + universe pipeline)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlm.execution.dte_utils import dte_from_plan

# Keep in sync with ``scripts/monitor_active_trade_plans._TRADE_LOG_COLUMNS``.
TRADE_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "plan_id",
    "symbol",
    "strategy",
    "entry_debit",
    "entry_mid",
    "current_mark",
    "peak_mark",
    "unrealized_pnl",
    "unrealized_pnl_pct",
    "signal",
    "closed",
    "dte",
    "legs_json",
)


def _migrate_columns(log_path: Path) -> None:
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_names = list(reader.fieldnames or [])
            if old_names and all(c in old_names for c in TRADE_LOG_COLUMNS):
                return
    except OSError:
        return
    rows: list[dict[str, str]] = []
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRADE_LOG_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: str(r.get(k, "")) for k in TRADE_LOG_COLUMNS})


def ensure_trade_log(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(TRADE_LOG_COLUMNS))


def open_plan_ids(path: Path) -> set[str]:
    """``plan_id`` values whose latest row has ``closed`` != 1."""
    if not path.is_file():
        return set()
    latest: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("plan_id") or "").strip()
                if pid:
                    latest[pid] = {k: str(v) for k, v in row.items()}
    except OSError:
        return set()
    return {pid for pid, row in latest.items() if str(row.get("closed") or "0").strip() != "1"}


def _legs_json(matched_legs: list[dict[str, Any]] | None) -> str:
    if not matched_legs:
        return ""
    slim: list[dict[str, Any]] = []
    for leg in matched_legs:
        if not isinstance(leg, dict):
            continue
        slim.append(
            {
                k: leg.get(k)
                for k in (
                    "symbol",
                    "contract_symbol",
                    "side",
                    "option_type",
                    "strike",
                    "expiry",
                    "bid",
                    "ask",
                    "mid",
                )
                if leg.get(k) is not None
            }
        )
    return json.dumps(slim, separators=(",", ":"), default=str)


def _strategy_name(plan: dict[str, Any]) -> str:
    dec = plan.get("decision")
    if isinstance(dec, dict):
        name = str(dec.get("strategy_name") or dec.get("strategy") or "").strip()
        if name:
            return name
    return str(plan.get("strategy") or plan.get("strategy_name") or "").strip()


def trade_log_row_from_active_plan(plan: dict[str, Any]) -> dict[str, str] | None:
    """Build an open ``trade_log`` row from a finalized active universe plan."""
    if str(plan.get("status") or "").strip().lower() != "active":
        return None
    pid = str(plan.get("plan_id") or "").strip()
    sym = str(plan.get("symbol") or "").strip().upper()
    if not pid or not sym:
        return None
    try:
        raw_debit = plan.get("entry_debit_dollars")
        entry_debit = float(0.0 if raw_debit is None else raw_debit)
        raw_mid = plan.get("entry_mid_mark_dollars")
        if raw_mid is None:
            entry_mid = entry_debit
        else:
            entry_mid = float(raw_mid)
    except (TypeError, ValueError):
        return None
    # Debit > 0, credit < 0 (see chain_match.estimate_entry_cost_from_matched_legs).
    # Reject only a zero/invalid book — negative credits must still seed the paper log.
    if abs(entry_debit) < 1e-9 and abs(entry_mid) < 1e-9:
        return None
    mark = entry_mid if abs(entry_mid) >= 1e-9 else entry_debit
    pnl = mark - entry_debit
    pnl_pct = (pnl / abs(entry_debit) * 100.0) if abs(entry_debit) > 1e-6 else 0.0
    plan_dte = dte_from_plan(plan)
    legs = list(plan.get("matched_legs") or [])
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "plan_id": pid,
        "symbol": sym,
        "strategy": _strategy_name(plan),
        "entry_debit": f"{round(entry_debit, 4)}",
        "entry_mid": f"{round(entry_mid, 4)}",
        "current_mark": f"{round(mark, 4)}",
        "peak_mark": f"{round(mark, 4)}",
        "unrealized_pnl": f"{round(pnl, 4)}",
        "unrealized_pnl_pct": f"{round(pnl_pct, 2)}",
        "signal": "hold",
        "closed": "0",
        "dte": f"{round(plan_dte, 3)}" if plan_dte == plan_dte else "",
        "legs_json": _legs_json(legs),
    }


def upsert_trade_log_row(path: Path, row: dict[str, str]) -> None:
    """Upsert one row per ``plan_id`` (latest state)."""
    ensure_trade_log(path)
    _migrate_columns(path)
    rows: dict[str, dict[str, str]] = {}
    if path.is_file() and path.stat().st_size > 0:
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                for existing in csv.DictReader(f):
                    pid = str(existing.get("plan_id") or "").strip()
                    if pid:
                        rows[pid] = {k: str(existing.get(k, "")) for k in TRADE_LOG_COLUMNS}
        except OSError:
            rows = {}
    pid = str(row.get("plan_id") or "").strip()
    if not pid:
        return
    merged = rows.get(pid, {})
    for col in TRADE_LOG_COLUMNS:
        if col in row and row[col] is not None:
            merged[col] = str(row[col])
    rows[pid] = merged
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRADE_LOG_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for r in rows.values():
            writer.writerow(r)


def close_stale_open_rows_above_dte(log_path: Path, *, max_dte: float = 5.0) -> int:
    """Mark open rows with DTE above ``max_dte`` as closed (legacy swing ghosts)."""
    if not log_path.is_file():
        return 0
    latest: dict[str, dict[str, str]] = {}
    try:
        with log_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("plan_id") or "").strip()
                if pid:
                    latest[pid] = {k: str(v) for k, v in row.items()}
    except OSError:
        return 0

    closed = 0
    now = datetime.now(timezone.utc).isoformat()
    for pid, row in latest.items():
        if str(row.get("closed") or "0").strip() == "1":
            continue
        try:
            dte = float(row.get("dte") or 0.0)
        except (TypeError, ValueError):
            dte = 0.0
        if dte <= max_dte:
            continue
        row = dict(row)
        row["timestamp_utc"] = now
        row["signal"] = "stale_dte_cleanup"
        row["closed"] = "1"
        latest[pid] = row
        closed += 1

    if closed:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(TRADE_LOG_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for r in latest.values():
                writer.writerow(r)
    return closed


def seed_paper_opens_from_active_plans(
    active_plans: list[dict[str, Any]],
    log_path: Path,
) -> list[str]:
    """Open paper rows for active plans not already open in ``log_path``. Returns new ``plan_id``s."""
    open_ids = open_plan_ids(log_path)
    seeded: list[str] = []
    for plan in active_plans:
        if not isinstance(plan, dict):
            continue
        pid = str(plan.get("plan_id") or "").strip()
        if not pid or pid in open_ids:
            continue
        row = trade_log_row_from_active_plan(plan)
        if row is None:
            continue
        upsert_trade_log_row(log_path, row)
        open_ids.add(pid)
        seeded.append(pid)
    return seeded
