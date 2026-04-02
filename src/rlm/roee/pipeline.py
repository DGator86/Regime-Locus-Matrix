from __future__ import annotations

import pandas as pd

from rlm.roee.policy import select_trade


def apply_roee_policy(df: pd.DataFrame, strike_increment: float = 1.0) -> pd.DataFrame:
    """
    Applies the ROEE decision policy row-by-row and stores summarized outputs.
    """
    required = [
        "close",
        "sigma",
        "S_D",
        "S_V",
        "S_L",
        "S_G",
        "direction_regime",
        "volatility_regime",
        "liquidity_regime",
        "dealer_flow_regime",
        "regime_key",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ROEE policy: {missing}")

    out = df.copy()

    actions = []
    strategy_names = []
    rationales = []
    size_fractions = []
    target_profit_pcts = []
    max_risk_pcts = []
    leg_counts = []

    for _, row in out.iterrows():
        decision = select_trade(
            current_price=float(row["close"]),
            sigma=float(row["sigma"]),
            s_d=float(row["S_D"]),
            s_v=float(row["S_V"]),
            s_l=float(row["S_L"]),
            s_g=float(row["S_G"]),
            direction_regime=str(row["direction_regime"]),
            volatility_regime=str(row["volatility_regime"]),
            liquidity_regime=str(row["liquidity_regime"]),
            dealer_flow_regime=str(row["dealer_flow_regime"]),
            regime_key=str(row["regime_key"]),
            bid_ask_spread_pct=(
                float(row["bid_ask_spread"] / row["close"])
                if "bid_ask_spread" in out.columns and pd.notna(row.get("bid_ask_spread"))
                else None
            ),
            has_major_event=(
                bool(row["has_major_event"])
                if "has_major_event" in out.columns and pd.notna(row.get("has_major_event"))
                else False
            ),
            strike_increment=strike_increment,
        )

        actions.append(decision.action)
        strategy_names.append(decision.strategy_name)
        rationales.append(decision.rationale)
        size_fractions.append(decision.size_fraction)
        target_profit_pcts.append(decision.target_profit_pct)
        max_risk_pcts.append(decision.max_risk_pct)
        leg_counts.append(len(decision.legs))

    out["roee_action"] = actions
    out["roee_strategy"] = strategy_names
    out["roee_rationale"] = rationales
    out["roee_size_fraction"] = size_fractions
    out["roee_target_profit_pct"] = target_profit_pcts
    out["roee_max_risk_pct"] = max_risk_pcts
    out["roee_leg_count"] = leg_counts

    return out
