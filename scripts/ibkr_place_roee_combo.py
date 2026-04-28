#!/usr/bin/env python3
"""
Submit a **multi-leg option combo** (BAG) to Interactive Brokers from a small JSON spec.

Default is **review-only** (``transmit=False``): order is created in TWS but not released
to the market until you transmit from TWS, unless you pass ``--transmit``.

**Paper:** TWS paper ``IBKR_PORT=7497`` (or Gateway ``4002``). **Live** ports ``7496`` /
``4001`` require ``--acknowledge-live``.

JSON shape::

    {
      "underlying": "SPY",
      "quantity": 1,
      "limit_price": 1.55,
      "legs": [
        {"side": "long", "option_type": "call", "strike": 455.0, "expiry": "2026-04-18"},
        {"side": "short", "option_type": "call", "strike": 460.0, "expiry": "2026-04-18"}
      ]
    }

Example::

    python scripts/ibkr_place_roee_combo.py --spec my_combo.json --limit-price 1.55
    python scripts/ibkr_place_roee_combo.py --spec my_combo.json --limit-price 1.55 --transmit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.execution.ibkr_combo_orders import (
    IBKRLegAction,
    IBKROptionLegSpec,
    expiry_iso_to_ib,
    load_ibkr_order_socket_config,
    option_type_to_ib_right,
    place_options_combo_limit_order,
    roee_side_to_ib_action,
)


def _parse_spec(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON file with underlying, legs[], optional quantity",
    )
    p.add_argument(
        "--limit-price",
        type=float,
        default=None,
        help='Net limit price (or set "limit_price" in JSON if omitted)',
    )
    p.add_argument("--quantity", type=int, default=None, help="Override JSON quantity (default 1)")
    p.add_argument(
        "--transmit",
        action="store_true",
        help="Release order to IBKR (otherwise TWS holds it for manual transmit)",
    )
    p.add_argument(
        "--acknowledge-live",
        action="store_true",
        help="Required when IBKR_PORT is a live session (7496 / 4001)",
    )
    p.add_argument(
        "--combo-action",
        choices=("BUY", "SELL"),
        default="BUY",
        help="Parent combo order side (BUY = typical net-debit structures)",
    )
    p.add_argument("--account", default=None, help="IB account code (sub-accounts / advisors)")
    p.add_argument("--tif", default="DAY", help="Time in force (default DAY)")
    args = p.parse_args()

    data = _parse_spec(args.spec)
    underlying = str(data.get("underlying", "")).upper().strip()
    if not underlying:
        print("spec must include 'underlying'", file=sys.stderr)
        return 2
    legs_raw = data.get("legs")
    if not isinstance(legs_raw, list) or not legs_raw:
        print("spec must include non-empty 'legs' array", file=sys.stderr)
        return 2

    qty = int(args.quantity if args.quantity is not None else data.get("quantity", 1))
    lim_raw = args.limit_price if args.limit_price is not None else data.get("limit_price")
    if lim_raw is None:
        print('Provide --limit-price or "limit_price" in the JSON spec.', file=sys.stderr)
        return 2
    limit_price = float(lim_raw)

    ib_legs: list[tuple[IBKROptionLegSpec, IBKRLegAction]] = []
    for i, leg in enumerate(legs_raw):
        if not isinstance(leg, dict):
            print(f"legs[{i}] must be an object", file=sys.stderr)
            return 2
        try:
            spec = IBKROptionLegSpec(
                underlying=underlying,
                expiry_yyyymmdd=expiry_iso_to_ib(str(leg["expiry"])),
                strike=float(leg["strike"]),
                right=option_type_to_ib_right(str(leg["option_type"])),
            )
            act = roee_side_to_ib_action(str(leg["side"]))
            ib_legs.append((spec, act))
        except (KeyError, ValueError, TypeError) as e:
            print(f"legs[{i}] invalid: {e}", file=sys.stderr)
            return 2

    h, port, cid = load_ibkr_order_socket_config()
    print(f"IBKR {h}:{port} clientId={cid}  transmit={args.transmit}  legs={len(ib_legs)}")
    try:
        oid, trail = place_options_combo_limit_order(
            ib_legs,
            quantity=qty,
            limit_price=limit_price,
            transmit=args.transmit,
            acknowledge_live=args.acknowledge_live,
            combo_order_action=args.combo_action,
            account=args.account,
            tif=args.tif,
        )
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print(f"orderId={oid}  status_trail={trail}")
    if not args.transmit:
        print("Note: transmit=False — complete or transmit from TWS if you want it working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
