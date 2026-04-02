from __future__ import annotations

import pandas as pd

from rlm.scoring.thresholds import (
    classify_dealer_flow,
    classify_direction,
    classify_liquidity,
    classify_volatility,
)


def make_regime_key(
    direction: str,
    volatility: str,
    liquidity: str,
    dealer_flow: str,
) -> str:
    return f"{direction}|{volatility}|{liquidity}|{dealer_flow}"


def classify_state_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires:
      S_D, S_V, S_L, S_G
    Produces:
      direction_regime, volatility_regime, liquidity_regime, dealer_flow_regime, regime_key
    """
    required = ["S_D", "S_V", "S_L", "S_G"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required score columns for state matrix classification: {missing}"
        )

    out = df.copy()

    out["direction_regime"] = out["S_D"].apply(classify_direction)
    out["volatility_regime"] = out["S_V"].apply(classify_volatility)
    out["liquidity_regime"] = out["S_L"].apply(classify_liquidity)
    out["dealer_flow_regime"] = out["S_G"].apply(classify_dealer_flow)

    out["regime_key"] = out.apply(
        lambda row: make_regime_key(
            row["direction_regime"],
            row["volatility_regime"],
            row["liquidity_regime"],
            row["dealer_flow_regime"],
        ),
        axis=1,
    )

    return out


def regime_is_tradeable(
    direction_regime: str,
    volatility_regime: str,
    liquidity_regime: str,
    dealer_flow_regime: str,
) -> bool:
    """
    Conservative first-pass tradeability filter.
    We do not trade ambiguous direction/volatility unknown states.
    """
    banned = {"unknown"}
    if (
        direction_regime in banned
        or volatility_regime in banned
        or liquidity_regime in banned
        or dealer_flow_regime in banned
    ):
        return False

    # permit transition states, but later policy can decide smaller size / skip
    return True
