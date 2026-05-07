"""Regime-to-strategy map for the $1K→$25K PDT Challenge aggressive sniper.

``STRATEGY_MAP_CHALLENGE`` maps a four-element regime tuple
    (direction, vol, liquidity, dealer_flow)
to the name of an aggressive day-trader candidate built by
``build_candidate_from_strategy_name`` in ``rlm.roee.policy``.

Regime label vocabulary matches the canonical RLM classifiers:
  direction  : "bull" | "bear" | "range" | "transition"
  vol        : "high_vol" | "low_vol" | "transition"
  liquidity  : "high_liquidity" | "low_liquidity"
  dealer_flow: "supportive" | "destabilizing"

Any regime not present in the map returns ``"no_trade"``.
"""

from __future__ import annotations

STRATEGY_MAP_CHALLENGE: dict[tuple[str, str, str, str], str] = {
    # Bullish momentum + cheap premium (low vol) → long call
    ("bull", "low_vol",  "high_liquidity", "supportive"):    "aggressive_daytrader_call",
    # Bullish + high vol → prefer ATM straddle to capture outsized move either way
    ("bull", "high_vol", "high_liquidity", "supportive"):    "aggressive_daytrader_0DTE_straddle",
    # Bearish momentum — supportive or destabilizing dealer flow both produce put setups
    ("bear", "low_vol",  "high_liquidity", "supportive"):    "aggressive_daytrader_put",
    ("bear", "high_vol", "high_liquidity", "supportive"):    "aggressive_daytrader_put",
    ("bear", "low_vol",  "high_liquidity", "destabilizing"): "aggressive_daytrader_put",
    ("bear", "high_vol", "high_liquidity", "destabilizing"): "aggressive_daytrader_put",
}


def get_challenge_strategy(regime_tuple: tuple[str, str, str, str]) -> str:
    """Return the aggressive strategy name for a regime tuple, or ``"no_trade"``."""
    return STRATEGY_MAP_CHALLENGE.get(regime_tuple, "no_trade")
