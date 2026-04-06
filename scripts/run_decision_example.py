#!/usr/bin/env python3
"""
Smoke the ROEE entry point (`select_trade`) with a single synthetic row — no CSV, no IBKR.

Run from repo root:

    python scripts/run_decision_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.roee.policy import select_trade


def main() -> int:
    # Same shape as one forecast row → policy inputs (see tests/unit/test_roee_policy.py)
    decision = select_trade(
        current_price=452.3,
        sigma=0.018,
        s_d=0.85,
        s_v=-0.3,
        s_l=0.6,
        s_g=0.4,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        regime_key="bull|low_vol|high_liquidity|supportive",
        bid_ask_spread_pct=0.02,
        has_major_event=False,
        strike_increment=0.5,
    )

    print("ACTION:", decision.action)
    print("STRATEGY:", decision.strategy_name)
    print("RATIONALE:", decision.rationale)
    print("SIZE FRACTION:", decision.size_fraction)
    print("TARGET PROFIT %:", decision.target_profit_pct)
    print("MAX RISK %:", decision.max_risk_pct)
    print("LEGS:")
    for leg in decision.legs or []:
        print(f"  {leg.side} {leg.option_type} @ {leg.strike}")
    if decision.metadata:
        print("METADATA:", {k: decision.metadata[k] for k in sorted(decision.metadata) if k != "confidence"})
        if "confidence" in decision.metadata:
            print("  confidence:", decision.metadata["confidence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
