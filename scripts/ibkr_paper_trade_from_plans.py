#!/usr/bin/env python3
"""
Submit **opening** combo limit orders to IBKR for each **active** plan in ``universe_trade_plans.json``.

**Paper only:** ``IBKR_PORT`` must be **7497** (TWS paper) or **4002** (Gateway paper).

Examples::

    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --dry-run
    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --max 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.execution.ibkr_combo_orders import (
    assert_paper_trading_port,
    legs_from_ibkr_combo_spec,
    load_ibkr_order_socket_config,
    place_options_combo_limit_order,
)


def _load_plans(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--max", type=int, default=20, help="Max opening orders (safety cap)")
    p.add_argument("--dry-run", action="store_true", help="Print only; no IBKR calls")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between orders")
    args = p.parse_args()

    plans_path = ROOT / args.plans if not args.plans.is_absolute() else args.plans
    if not plans_path.is_file():
        print(f"Missing {plans_path}", file=sys.stderr)
        return 1

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

    n = 0
    for row in active[: max(0, args.max)]:
        spec = row.get("ibkr_combo_spec")
        if not isinstance(spec, dict):
            print(f"SKIP {row.get('symbol')}: no ibkr_combo_spec", file=sys.stderr)
            continue
        sym = row.get("symbol", "?")
        try:
            legs = legs_from_ibkr_combo_spec(spec)
        except Exception as e:
            print(f"SKIP {sym}: {e}", file=sys.stderr)
            continue
        qty = int(spec.get("quantity", 1))
        lim = float(spec.get("limit_price", 0))
        if lim <= 0:
            print(f"SKIP {sym}: bad limit_price", file=sys.stderr)
            continue

        combo = str(spec.get("combo_order_action", "BUY")).upper()
        if combo not in ("BUY", "SELL"):
            combo = "BUY"

        if args.dry_run:
            print(f"DRY-RUN {sym} qty={qty} {combo} LMT {lim} legs={len(legs)}")
            n += 1
            continue

        try:
            oid, trail = place_options_combo_limit_order(
                legs,
                quantity=qty,
                limit_price=lim,
                transmit=True,
                acknowledge_live=False,
                combo_order_action=combo,  # type: ignore[arg-type]
            )
            print(f"PLACED {sym} orderId={oid} trail={trail}")
            n += 1
        except Exception as e:
            print(f"FAIL {sym}: {e}", file=sys.stderr)
        time.sleep(max(0.0, args.delay))

    print(f"Done. Submitted (or dry-listed): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
