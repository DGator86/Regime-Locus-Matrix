#!/usr/bin/env python3
"""
Submit **opening** combo limit orders to IBKR for each **active** plan in ``universe_trade_plans.json``.

**Paper only:** ``IBKR_PORT`` must be **7497** (TWS paper) or **4002** (Gateway paper).

Uses a **single** IBKR connection for all orders (avoids repeated connect/disconnect and
``client id already in use`` (326) errors from stale TWS slots).

On successful placement, marks each plan with ``paper_opened: true`` inside the plans JSON so
the duplicate-guard in ``run_universe_options_pipeline.py`` can distinguish confirmed positions
from hypothetical / unentered plans.

Examples::

    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --dry-run
    python scripts/ibkr_paper_trade_from_plans.py --plans data/processed/universe_trade_plans.json --max 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.execution.ibkr_combo_orders import (
    assert_paper_trading_port,
    ibkr_order_connection,
    legs_from_ibkr_combo_spec,
    load_ibkr_order_socket_config,
)
from rlm.roee.system_gate import SystemGate


def _env_truthy(key: str) -> bool:
    v = (os.environ.get(key) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _load_plans(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_plans(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _mark_opened(plans_path: Path, plan_id: str) -> None:
    """Set paper_opened=true on the matching plan inside the JSON file."""
    try:
        payload = _load_plans(plans_path)
        changed = False
        for key in ("active_ranked", "results"):
            for row in payload.get(key) or []:
                if str(row.get("plan_id") or row.get("symbol")) == plan_id:
                    row["paper_opened"] = True
                    changed = True
        if changed:
            _save_plans(plans_path, payload)
    except Exception as e:
        print(f"[warn] could not mark paper_opened for {plan_id}: {e}", file=sys.stderr)


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

    # Filter to plans not yet confirmed opened (avoid re-opening)
    to_open = [r for r in active[: max(0, args.max)] if not r.get("paper_opened")]

    if not to_open:
        print("Done. Submitted (or dry-listed): 0")
        return 0

    if args.dry_run:
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

    # --- Single shared IBKR connection for all orders ---
    n = 0
    host, port, client_id = load_ibkr_order_socket_config()
    try:
        with ibkr_order_connection(host=host, port=port, client_id=client_id, timeout_sec=60.0) as app:
            for row in to_open:
                spec = row.get("ibkr_combo_spec")
                if not isinstance(spec, dict):
                    print(f"SKIP {row.get('symbol')}: no ibkr_combo_spec", file=sys.stderr)
                    continue
                sym = row.get("symbol", "?")
                plan_id = str(row.get("plan_id") or sym)
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

                try:
                    from rlm.execution.ibkr_combo_orders import (
                        _get_bundle,
                        _resolve_option_on_app,
                        _wait_order_terminal,
                        assert_paper_or_live_acknowledged,
                    )

                    _, _, Contract, ComboLeg, Order = _get_bundle()

                    assert_paper_or_live_acknowledged(port, acknowledge_live=False)

                    combo_contracts = []
                    und0 = legs[0][0].underlying.upper()
                    for leg_spec, leg_action in legs:
                        rc = _resolve_option_on_app(app, Contract, leg_spec, timeout_sec=45.0)
                        combo_contracts.append((rc, leg_action))

                    bag = Contract()
                    bag.symbol = und0
                    bag.secType = "BAG"
                    bag.currency = legs[0][0].currency
                    bag.exchange = "SMART"
                    bag.comboLegs = []
                    for rc, act in combo_contracts:
                        cl = ComboLeg()
                        cl.conId = int(rc.conId)
                        cl.ratio = 1
                        cl.action = act
                        cl.exchange = "SMART"
                        bag.comboLegs.append(cl)

                    if app._next_order_id is None:
                        raise RuntimeError("IBKR did not provide nextValidId")
                    oid = int(app._next_order_id)
                    app._next_order_id = oid + 1

                    order = Order()
                    order.action = combo
                    order.totalQuantity = float(qty)
                    order.orderType = "LMT"
                    order.lmtPrice = float(lim)
                    order.transmit = True
                    order.tif = "DAY"
                    order.eTradeOnly = False
                    order.firmQuoteOnly = False

                    app._order_status.pop(oid, None)
                    app.placeOrder(oid, bag, order)
                    trail = _wait_order_terminal(app, oid, transmit=True, timeout_sec=90.0)
                    print(f"PLACED {sym} orderId={oid} trail={trail}")
                    _mark_opened(plans_path, plan_id)
                    n += 1

                except Exception as e:
                    print(f"FAIL {sym}: {e}", file=sys.stderr)
                time.sleep(max(0.0, args.delay))
    except Exception as e:
        print(f"IBKR connection failed: {e}", file=sys.stderr)

    print(f"Done. Submitted (or dry-listed): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
