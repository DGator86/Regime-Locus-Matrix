#!/usr/bin/env python3
"""Compact options monitor CSV to one row per plan_id (repairs multi-GB append logs)."""

from __future__ import annotations

import argparse
import csv
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


def compact(path: Path) -> tuple[int, int]:
    before = path.stat().st_size if path.is_file() else 0
    latest = _latest_by_plan(path)
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
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    for cand in options_trade_log_read_paths(root):
        if not cand.is_file():
            continue
        before, after = compact(cand)
        print(f"Compacted {cand}: {before:,} bytes -> {after:,} bytes ({len(_latest_by_plan(cand))} plans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
