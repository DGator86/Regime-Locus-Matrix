#!/usr/bin/env python3
"""Compact options monitor CSV to one row per plan_id (repairs multi-GB append logs)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rlm.notify.options_paths import options_trade_log_read_paths  # noqa: E402

_TRADE_LOG_COLUMNS = [
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
]


def _latest_by_plan(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("plan_id") or "").strip()
            if pid:
                rows[pid] = {k: str(row.get(k, "")) for k in _TRADE_LOG_COLUMNS}
    return rows


def _active_plan_ids_from_universe(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    ids: set[str] = set()
    for row in payload.get("active_ranked") or []:
        if isinstance(row, dict):
            pid = str(row.get("plan_id") or "").strip()
            if pid:
                ids.add(pid)
    for row in payload.get("results") or []:
        if isinstance(row, dict) and str(row.get("status") or "") == "active":
            pid = str(row.get("plan_id") or "").strip()
            if pid:
                ids.add(pid)
    return ids


def _close_open_not_in_universe(
    rows: dict[str, dict[str, str]],
    *,
    active_ids: set[str],
) -> dict[str, dict[str, str]]:
    for pid, row in list(rows.items()):
        if (row.get("closed") or "0").strip() == "1":
            continue
        if pid in active_ids:
            continue
        updated = dict(row)
        updated["closed"] = "1"
        updated["signal"] = updated.get("signal") or "stale_not_in_universe"
        rows[pid] = updated
    return rows


def _dedupe_open_per_symbol(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Keep one newest open row per symbol; mark older duplicates closed."""
    open_rows = [
        (pid, row)
        for pid, row in rows.items()
        if (row.get("closed") or "0").strip() != "1"
    ]
    by_sym: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for pid, row in open_rows:
        sym = str(row.get("symbol") or "").strip().upper()
        if sym:
            by_sym.setdefault(sym, []).append((pid, row))
    for sym, items in by_sym.items():
        if len(items) <= 1:
            continue
        items.sort(key=lambda t: str(t[1].get("timestamp_utc") or ""), reverse=True)
        for pid, row in items[1:]:
            row = dict(row)
            row["closed"] = "1"
            row["signal"] = row.get("signal") or "dedupe_closed"
            rows[pid] = row
    return rows


def compact(
    path: Path,
    *,
    one_open_per_symbol: bool = False,
    close_not_in_universe: Path | None = None,
) -> tuple[int, int]:
    before = path.stat().st_size if path.is_file() else 0
    latest = _latest_by_plan(path)
    if close_not_in_universe is not None:
        latest = _close_open_not_in_universe(
            latest,
            active_ids=_active_plan_ids_from_universe(close_not_in_universe),
        )
    if one_open_per_symbol:
        latest = _dedupe_open_per_symbol(latest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_TRADE_LOG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in latest.values():
            w.writerow(row)
    after = path.stat().st_size
    return before, after


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(os.environ.get("RLM_ROOT", str(REPO_ROOT))))
    p.add_argument(
        "--one-open-per-symbol",
        action="store_true",
        help="Mark duplicate open rows per symbol as closed (keeps newest timestamp only)",
    )
    p.add_argument(
        "--close-not-in-universe",
        action="store_true",
        help="Mark open rows closed when plan_id is not in universe_trade_plans.json actives",
    )
    p.add_argument(
        "--universe",
        type=Path,
        default=Path("data/processed/universe_trade_plans.json"),
        help="Universe JSON for --close-not-in-universe",
    )
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    universe_path = (
        args.universe if args.universe.is_absolute() else (root / args.universe)
    ).resolve()
    close_uni = universe_path if args.close_not_in_universe else None
    for cand in options_trade_log_read_paths(root):
        if not cand.is_file():
            continue
        before, after = compact(
            cand,
            one_open_per_symbol=bool(args.one_open_per_symbol),
            close_not_in_universe=close_uni,
        )
        open_n = sum(
            1
            for row in _latest_by_plan(cand).values()
            if (row.get("closed") or "0").strip() != "1"
        )
        print(
            f"Compacted {cand}: {before:,} bytes -> {after:,} bytes "
            f"({len(_latest_by_plan(cand))} plans, {open_n} open)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
