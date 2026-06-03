#!/usr/bin/env python3
"""
Open local paper options rows from ``universe_trade_plans.json`` (same as Robinhood alerts).

Writes/updates ``trade_log.csv`` via :func:`rlm.execution.trade_log_io.seed_paper_opens_from_active_plans`.
Does **not** send orders to IBKR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.execution.trade_log_io import ensure_trade_log, open_plan_ids, seed_paper_opens_from_active_plans
from rlm.roee.system_gate import SystemGate


def _env_truthy(key: str) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _load_plans(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_plans(payload: dict) -> list[dict]:
    ranked = list(payload.get("active_ranked") or [])
    if ranked:
        return [r for r in ranked if isinstance(r, dict) and r.get("status") == "active"]
    return [r for r in payload.get("results", []) if isinstance(r, dict) and r.get("status") == "active"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument(
        "--trade-log",
        type=Path,
        default=Path("data/processed/trade_log.csv"),
        help="Paper monitor CSV (default: data/processed/trade_log.csv)",
    )
    p.add_argument("--max", type=int, default=20, help="Max active plans to consider (safety cap)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List would-open plans only; do not write trade_log",
    )
    args = p.parse_args()

    data_root = Path(os.environ.get("RLM_ROOT", str(ROOT))).expanduser().resolve()
    plans_path = args.plans if args.plans.is_absolute() else data_root / args.plans
    log_path = args.trade_log if args.trade_log.is_absolute() else data_root / args.trade_log

    if not plans_path.is_file():
        print(f"Missing {plans_path}", file=sys.stderr)
        return 1

    gate = SystemGate(data_root)
    if _env_truthy("RLM_SKIP_SYSTEM_GATE"):
        gate_allowed, gs = True, gate.load()
        print("[paper-trade] RLM_SKIP_SYSTEM_GATE=1 — ignoring system gate for this run", flush=True)
    else:
        gate_allowed, gs = gate.check()
    if not gate_allowed:
        print(
            f"[paper-trade] trading paused by system gate — posture={gs.posture} status={gs.status}",
            flush=True,
        )
        return 0

    payload = _load_plans(plans_path)
    active = _active_plans(payload)[: max(0, args.max)]
    already = open_plan_ids(log_path)

    if args.dry_run:
        would = [
            str(r.get("plan_id") or "")
            for r in active
            if str(r.get("plan_id") or "").strip() and str(r.get("plan_id")) not in already
        ]
        for pid in would:
            sym = next((r.get("symbol") for r in active if r.get("plan_id") == pid), "?")
            print(f"DRY-RUN would open {sym} {pid} in {log_path}")
        print(f"Done. Would open: {len(would)}")
        return 0

    ensure_trade_log(log_path)
    seeded = seed_paper_opens_from_active_plans(active, log_path)
    print(f"Done. Opened in trade_log: {len(seeded)}  ({', '.join(seeded) if seeded else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
