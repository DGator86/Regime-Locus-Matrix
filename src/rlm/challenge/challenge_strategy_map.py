"""Regime-to-strategy map for the $1K→$25K PDT Challenge aggressive sniper.

``STRATEGY_MAP_CHALLENGE`` maps a four-element regime tuple
    (direction, vol, liquidity, dealer_flow)
to the name of an aggressive day-trader candidate built by
``build_candidate_from_strategy_name`` in ``rlm.roee.policy``.

Any regime not present in the map returns ``"no_trade"``.
"""

from __future__ import annotations

STRATEGY_MAP_CHALLENGE: dict[tuple[str, str, str, str], str] = {
    # Bullish momentum + cheap premium → long call
    ("bullish", "low_vol",    "liquid", "buying_flow"): "aggressive_daytrader_call",
    ("bullish", "medium_vol", "liquid", "buying_flow"): "aggressive_daytrader_call",
    # Bullish + high vol → prefer ATM straddle to capture outsized move either way
    ("bullish", "high_vol",   "liquid", "buying_flow"): "aggressive_daytrader_0DTE_straddle",
    # Bearish momentum → long put across all vol levels
    ("bearish", "high_vol",   "liquid", "selling_flow"): "aggressive_daytrader_put",
    ("bearish", "medium_vol", "liquid", "selling_flow"): "aggressive_daytrader_put",
    ("bearish", "low_vol",    "liquid", "selling_flow"): "aggressive_daytrader_put",
}


def get_challenge_strategy(regime_tuple: tuple[str, str, str, str]) -> str:
    """Return the aggressive strategy name for a regime tuple, or ``"no_trade"``."""
    return STRATEGY_MAP_CHALLENGE.get(regime_tuple, "no_trade")
