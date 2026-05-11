#!/usr/bin/env python3
"""
**Dry-run:** print intended multi-leg option opens from ``universe_trade_plans.json``.

IBKR in RLM is **equities-only**. Without ``--dry-run`` this script exits immediately.
``IBKR_PORT`` is validated as **paper** for consistency with the rest of the stack (equity scripts).

Examples::

    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --dry-run
    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --dry-run --max 3
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

from rlm.execution.ibkr_combo_orders import (
    assert_paper_trading_port,
    legs_from_ibkr_combo_spec,
    load_ibkr_order_socket_config,
)
from rlm.roee.system_gate import SystemGate


def _env_truthy(key: str) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _load_plans(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--max", type=int, default=20, help="Max plans to list (safety cap)")
    p.add_argument("--dry-run", action="store_true", help="Required: print combos only")
    args = p.parse_args()

    if not args.dry_run:
        from rlm.execution.options_ibkr_policy import exit_ibkr_option_combo_blocked

        exit_ibkr_option_combo_blocked(
            "ibkr_paper_trade_from_plans without --dry-run (IBKR is equities-only)"
        )

    plans_path = ROOT / args.plans if not args.plans.is_absolute() else args.plans
    if not plans_path.is_file():
        print(f"Missing {plans_path}", file=sys.stderr)
        return 1

    gate = SystemGate(ROOT)
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

    _, port, _ = load_ibkr_order_socket_config()
    try:
        assert_paper_trading_port(port)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    payload = _load_plans(plans_path)
    ranked = list(payload.get("active_ranked") or [])
    if ranked:
        active = ranked
    else:
        active = [r for r in payload.get("results", []) if r.get("status") == "active"]

    to_open = [r for r in active[: max(0, args.max)] if not r.get("paper_opened")]

    if not to_open:
        print("Done. Submitted (or dry-listed): 0")
        return 0

    n = 0
    for row in to_open:
        spec = row.get("ibkr_combo_spec")
        if not isinstance(spec, dict):
            print(f"SKIP {row.get('symbol')}: no ibkr_combo_spec", file=sys.stderr)
            continue
        sym = row.get("symbol", "?")
        qty = int(spec.get("quantity", 1))
        lim = float(spec.get("limit_price", 0))
        combo = str(spec.get("combo_order_action", "BUY")).upper()
        try:
            legs = legs_from_ibkr_combo_spec(spec)
        except Exception as e:
            print(f"SKIP {sym}: {e}", file=sys.stderr)
            continue
        if lim <= 0:
            print(f"SKIP {sym}: bad limit_price", file=sys.stderr)
            continue
        print(f"DRY-RUN {sym} qty={qty} {combo} LMT {lim} legs={len(legs)}")
        n += 1
    print(f"Done. Submitted (or dry-listed): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
