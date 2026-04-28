#!/usr/bin/env python3
"""
Run ``select_trade`` then ``match_legs_to_chain`` on a normalized option-chain CSV/Parquet.

Chain file must include columns:
  timestamp, underlying, expiry, option_type, strike, bid, ask
(see ``rlm.data.option_chain.normalize_option_chain``).

Example:

    python scripts/run_decision_with_chain.py \\
      --chain data/raw/option_chain_SPY.csv \\
      --underlying SPY \\
      --close 452.3 --sigma 0.018 \\
      --direction bull --vol low_vol --liquidity high_liquidity --dealer supportive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.option_chain import normalize_option_chain, select_nearest_expiry_slice
from rlm.roee.chain_match import estimate_entry_cost_from_matched_legs, match_legs_to_chain
from rlm.roee.policy import select_trade


def _load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf in {".csv"}:
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported chain format: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain", type=Path, required=True, help="CSV or Parquet option chain")
    p.add_argument("--underlying", default="SPY")
    p.add_argument("--close", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--s-d", type=float, default=0.85, dest="s_d")
    p.add_argument("--s-v", type=float, default=-0.3, dest="s_v")
    p.add_argument("--s-l", type=float, default=0.6, dest="s_l")
    p.add_argument("--s-g", type=float, default=0.4, dest="s_g")
    p.add_argument("--direction", default="bull", choices=("bull", "bear", "range", "transition"))
    p.add_argument("--vol", default="low_vol", dest="volatility_regime")
    p.add_argument("--liquidity", default="high_liquidity", dest="liquidity_regime")
    p.add_argument("--dealer", default="supportive", dest="dealer_flow_regime")
    p.add_argument("--bid-ask-spread-pct", type=float, default=0.02, dest="bid_ask_spread_pct")
    p.add_argument("--strike-increment", type=float, default=0.5)
    p.add_argument("--no-expiry-slice", action="store_true", help="Use full chain (all expiries) for matching")
    p.add_argument(
        "--write-ibkr-spec",
        type=Path,
        default=None,
        help="After a successful match, write JSON for scripts/ibkr_place_roee_combo.py",
    )
    args = p.parse_args()

    u = args.underlying.upper()
    dr = str(args.direction)
    regime_key = f"{dr}|{args.volatility_regime}|{args.liquidity_regime}|{args.dealer_flow_regime}"

    raw = _load_table(args.chain)
    raw = raw[raw["underlying"].astype(str).str.upper() == u].copy()
    if raw.empty:
        print(f"No rows for underlying={u!r} in {args.chain}", file=sys.stderr)
        return 1

    chain = normalize_option_chain(raw)

    decision = select_trade(
        current_price=args.close,
        sigma=args.sigma,
        s_d=args.s_d,
        s_v=args.s_v,
        s_l=args.s_l,
        s_g=args.s_g,
        direction_regime=dr,
        volatility_regime=args.volatility_regime,
        liquidity_regime=args.liquidity_regime,
        dealer_flow_regime=args.dealer_flow_regime,
        regime_key=regime_key,
        bid_ask_spread_pct=args.bid_ask_spread_pct,
        has_major_event=False,
        strike_increment=args.strike_increment,
    )

    print("--- select_trade ---")
    print("action:", decision.action)
    print("strategy:", decision.strategy_name)
    print("rationale:", decision.rationale)
    print("size_fraction:", decision.size_fraction)
    for leg in decision.legs or []:
        print(f"  leg: {leg.side} {leg.option_type} @ {leg.strike}")

    if decision.action != "enter" or not decision.candidate:
        return 0

    slice_df = chain
    if not args.no_expiry_slice:
        c = decision.candidate
        slice_df = select_nearest_expiry_slice(chain, c.target_dte_min, c.target_dte_max)
        if slice_df.empty:
            print(
                "\n--- match_legs_to_chain ---\nSKIP: no contracts in DTE range "
                f"[{c.target_dte_min}, {c.target_dte_max}]",
                file=sys.stderr,
            )
            return 2

    matched = match_legs_to_chain(decision=decision, chain_slice=slice_df)
    print("\n--- match_legs_to_chain ---")
    print("action:", matched.action)
    print("rationale:", matched.rationale)
    if matched.action == "enter" and matched.metadata.get("matched_legs"):
        for m in matched.metadata["matched_legs"]:
            print(
                f"  matched: {m['side']} {m['option_type']} K={m['strike']} "
                f"exp={m['expiry']} bid={m['bid']:.2f} ask={m['ask']:.2f}"
            )
        debit = estimate_entry_cost_from_matched_legs(matched)
        if debit == debit:  # not NaN
            print(f"estimated entry debit (×100): {debit:.2f}")

        if args.write_ibkr_spec:
            mlegs = matched.metadata.get("matched_legs") or []
            lim = float(round(debit, 4)) if debit == debit else 0.0
            payload = {
                "underlying": u,
                "quantity": 1,
                "limit_price": lim,
                "legs": [
                    {
                        "side": str(m["side"]),
                        "option_type": str(m["option_type"]),
                        "strike": float(m["strike"]),
                        "expiry": str(m["expiry"]),
                    }
                    for m in mlegs
                ],
            }
            path = args.write_ibkr_spec
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote IBKR combo spec: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
