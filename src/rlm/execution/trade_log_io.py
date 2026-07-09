"""Shared options paper ``trade_log.csv`` I/O (monitor + universe pipeline)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlm.execution.dte_utils import dte_from_plan

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

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


@contextmanager
def _trade_log_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_trade_log_rows_atomic(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(TRADE_LOG_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: str(r.get(k, "")) for k in TRADE_LOG_COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _ensure_trade_log_unlocked(path: Path) -> None:
    if path.is_file():
        return
    _write_trade_log_rows_atomic(path, [])


def _migrate_columns_unlocked(log_path: Path) -> None:
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
    _write_trade_log_rows_atomic(log_path, rows)


def _migrate_columns(log_path: Path) -> None:
    with _trade_log_lock(log_path):
        _migrate_columns_unlocked(log_path)


def ensure_trade_log(path: Path) -> None:
    with _trade_log_lock(path):
        _ensure_trade_log_unlocked(path)


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
        entry_debit = float(plan.get("entry_debit_dollars") or 0.0)
        entry_mid = float(plan.get("entry_mid_mark_dollars") or entry_debit or 0.0)
    except (TypeError, ValueError):
        return None
    if entry_debit <= 0 and entry_mid <= 0:
        return None
    mark = entry_mid if entry_mid > 0 else entry_debit
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
    pid = str(row.get("plan_id") or "").strip()
    if not pid:
        return
    with _trade_log_lock(path):
        _ensure_trade_log_unlocked(path)
        _migrate_columns_unlocked(path)
        rows: dict[str, dict[str, str]] = {}
        if path.is_file() and path.stat().st_size > 0:
            try:
                with path.open("r", encoding="utf-8", newline="") as f:
                    for existing in csv.DictReader(f):
                        existing_pid = str(existing.get("plan_id") or "").strip()
                        if existing_pid:
                            rows[existing_pid] = {k: str(existing.get(k, "")) for k in TRADE_LOG_COLUMNS}
            except OSError:
                rows = {}
        merged = rows.get(pid, {})
        for col in TRADE_LOG_COLUMNS:
            if col in row and row[col] is not None:
                merged[col] = str(row[col])
        rows[pid] = merged
        _write_trade_log_rows_atomic(path, rows.values())


def close_stale_open_rows_above_dte(log_path: Path, *, max_dte: float = 5.0) -> int:
    """Mark open rows with DTE above ``max_dte`` as closed (legacy swing ghosts)."""
    with _trade_log_lock(log_path):
        if not log_path.is_file():
            return 0
        _migrate_columns_unlocked(log_path)
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
            _write_trade_log_rows_atomic(log_path, latest.values())
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
