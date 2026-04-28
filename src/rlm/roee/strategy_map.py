from __future__ import annotations

from rlm.types.options import TradeCandidate


def _candidate(
    strategy_name: str,
    regime_key: str,
    rationale: str,
    target_dte_min: int,
    target_dte_max: int,
    target_profit_pct: float,
    max_risk_pct: float,
    wings_sigma_low: float | None = None,
    wings_sigma_high: float | None = None,
    long_sigma: float | None = None,
    short_sigma: float | None = None,
    hedge_sigma: float | None = None,
    defined_risk: bool = True,
) -> TradeCandidate:
    return TradeCandidate(
        strategy_name=strategy_name,
        regime_key=regime_key,
        rationale=rationale,
        target_dte_min=target_dte_min,
        target_dte_max=target_dte_max,
        target_profit_pct=target_profit_pct,
        max_risk_pct=max_risk_pct,
        wings_sigma_low=wings_sigma_low,
        wings_sigma_high=wings_sigma_high,
        long_sigma=long_sigma,
        short_sigma=short_sigma,
        hedge_sigma=hedge_sigma,
        defined_risk=defined_risk,
    )


def _short_dte_strategy(
    direction: str,
    volatility: str,
    liquidity: str,
    dealer_flow: str,
    regime_key: str,
) -> TradeCandidate | None:
    """Return a 0–5 DTE intraday strategy candidate, or ``None`` if no match.

    These strategies target same-day and next-day expiry contracts (0DTE / 1DTE)
    on highly liquid underlyings (SPY, QQQ, SPX).  Risk per trade is kept tight
    because gamma exposure accelerates dramatically at low DTE.

    Strategy selection rules
    ------------------------
    bull   → 0DTE bull call spread (defined risk, fast delta capture)
    bear   → 0DTE bear put spread  (defined risk, fast delta capture)
    range + low_vol + supportive  → 0DTE iron condor (premium harvest)
    range + low_vol + destabilizing → 1DTE iron condor (small buffer)
    range + high_vol  → scalp long straddle (convexity; own the move)
    transition / high_vol → scalp long straddle (ambiguous → own gamma)
    transition / low_vol  → no-trade (ambiguous + low premium = bad risk/reward)
    """
    # ------------------------------------------------------------------ bull
    if direction == "bull":
        return _candidate(
            "0dte_bull_call_spread",
            regime_key,
            "Intraday bull regime; tight 0DTE call debit spread for fast delta capture.",
            0,
            2,
            target_profit_pct=0.40,
            max_risk_pct=0.01,
            long_sigma=0.25,
            short_sigma=0.75,
            defined_risk=True,
        )
    # ------------------------------------------------------------------ bear
    if direction == "bear":
        return _candidate(
            "0dte_bear_put_spread",
            regime_key,
            "Intraday bear regime; tight 0DTE put debit spread for fast delta capture.",
            0,
            2,
            target_profit_pct=0.40,
            max_risk_pct=0.01,
            long_sigma=-0.25,
            short_sigma=-0.75,
            defined_risk=True,
        )
    # ------------------------------------------------------------------ range
    if direction == "range":
        if volatility == "high_vol":
            return _candidate(
                "scalp_long_straddle",
                regime_key,
                "Range but vol elevated intraday; own the move with a short-dated straddle.",
                0,
                2,
                target_profit_pct=0.35,
                max_risk_pct=0.01,
                long_sigma=0.0,
                hedge_sigma=0.0,
                defined_risk=True,
            )
        if dealer_flow == "supportive":
            return _candidate(
                "0dte_iron_condor",
                regime_key,
                "Range + low vol + supportive flow; harvest same-day premium with a 0DTE iron condor.",
                0,
                2,
                target_profit_pct=0.50,
                max_risk_pct=0.01,
                wings_sigma_low=1.0,
                wings_sigma_high=1.5,
                defined_risk=True,
            )
        return _candidate(
            "1dte_iron_condor",
            regime_key,
            "Range + low vol + unstable flow; prefer next-day expiry for a small DTE buffer.",
            1,
            5,
            target_profit_pct=0.45,
            max_risk_pct=0.01,
            wings_sigma_low=1.0,
            wings_sigma_high=1.5,
            defined_risk=True,
        )
    # ------------------------------------------------------------------ transition (ambiguous direction)
    if volatility == "high_vol":
        return _candidate(
            "scalp_long_straddle",
            regime_key,
            "Ambiguous direction but high intraday vol; own gamma with a short-dated straddle.",
            0,
            2,
            target_profit_pct=0.30,
            max_risk_pct=0.008,
            long_sigma=0.0,
            hedge_sigma=0.0,
            defined_risk=True,
        )
    # transition + low vol → no clear edge intraday
    return None


def get_strategy_for_regime(
    direction: str,
    volatility: str,
    liquidity: str,
    dealer_flow: str,
    *,
    short_dte: bool = False,
) -> TradeCandidate:
    regime_key = f"{direction}|{volatility}|{liquidity}|{dealer_flow}"

    # Short-DTE (0DTE / 1DTE) intraday strategies take priority when requested.
    if short_dte:
        candidate = _short_dte_strategy(direction, volatility, liquidity, dealer_flow, regime_key)
        if candidate is not None:
            return candidate
        # Fall through to no_trade if short-DTE returns None (transition + low vol)
        return _candidate(
            "no_trade_or_micro_position",
            regime_key,
            "Ambiguous direction in short-DTE mode; skip to avoid low-premium gamma trap.",
            0,
            2,
            0.20,
            0.005,
            defined_risk=True,
        )

    # -----------------------------------------------------------------------
    # All strategies below are IBKR Level 2 compliant:
    # long single-leg options, long debit spreads, long straddles/strangles,
    # protective positions.  No credit spreads, no naked short legs.
    # -----------------------------------------------------------------------

    # Bull states
    if direction == "bull":
        if (
            volatility == "low_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            # Long call spread — Level 2 debit spread
            return _candidate(
                "long_call_spread",
                regime_key,
                "Bullish regime with supportive flow; long call debit spread (L2).",
                20,
                45,
                0.50,
                0.03,
                long_sigma=0.5,
                short_sigma=1.5,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            # Simple long call — less complex, Level 2
            return _candidate(
                "long_call",
                regime_key,
                "Bullish but flow less stable; plain long call for directional exposure (L2).",
                20,
                45,
                0.50,
                0.025,
                long_sigma=0.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            # Long call spread still valid in high vol — pay defined debit
            return _candidate(
                "long_call_spread",
                regime_key,
                "Bullish expansion with high liquidity; long call spread caps premium paid (L2).",
                14,
                35,
                0.45,
                0.025,
                long_sigma=0.5,
                short_sigma=1.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "low_liquidity"
            and dealer_flow == "destabilizing"
        ):
            # Long straddle — own convexity in both directions, Level 2
            return _candidate(
                "long_straddle",
                regime_key,
                "Bullish view but chaotic; own the move with a long straddle (L2).",
                14,
                30,
                0.35,
                0.02,
                long_sigma=0.0,
                hedge_sigma=0.0,
                defined_risk=True,
            )
        # Default bull: long call spread
        return _candidate(
            "long_call_spread",
            regime_key,
            "Default bullish defined-risk debit spread (L2).",
            20,
            45,
            0.45,
            0.02,
            long_sigma=0.5,
            short_sigma=1.5,
            defined_risk=True,
        )

    # Bear states
    if direction == "bear":
        if (
            volatility == "low_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            # Long put spread — Level 2 debit spread
            return _candidate(
                "long_put_spread",
                regime_key,
                "Bearish regime with supportive flow; long put debit spread (L2).",
                20,
                45,
                0.50,
                0.03,
                long_sigma=-0.5,
                short_sigma=-1.5,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            # Simple long put — Level 2
            return _candidate(
                "long_put",
                regime_key,
                "Bearish but less orderly; plain long put for directional exposure (L2).",
                20,
                45,
                0.50,
                0.025,
                long_sigma=-0.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            return _candidate(
                "long_put_spread",
                regime_key,
                "Bearish expansion with high liquidity; long put spread caps premium paid (L2).",
                14,
                35,
                0.45,
                0.025,
                long_sigma=-0.5,
                short_sigma=-1.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "low_liquidity"
            and dealer_flow == "destabilizing"
        ):
            return _candidate(
                "long_straddle",
                regime_key,
                "Bearish but chaotic and illiquid; own the move with a long straddle (L2).",
                14,
                30,
                0.35,
                0.02,
                long_sigma=0.0,
                hedge_sigma=0.0,
                defined_risk=True,
            )
        # Default bear: long put spread
        return _candidate(
            "long_put_spread",
            regime_key,
            "Default bearish defined-risk debit spread (L2).",
            20,
            45,
            0.45,
            0.02,
            long_sigma=-0.5,
            short_sigma=-1.5,
            defined_risk=True,
        )

    # Range states — no credit income strategies; use long strangles/straddles (Level 2)
    if direction == "range":
        if volatility == "low_vol" and dealer_flow == "supportive":
            # Long iron condor (debit) — Level 2; profits when stock moves significantly
            return _candidate(
                "long_iron_condor",
                regime_key,
                "Range with low vol; long iron condor debit structure owns breakout (L2).",
                20,
                45,
                0.40,
                0.02,
                wings_sigma_low=1.0,
                wings_sigma_high=1.5,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "long_strangle",
                regime_key,
                "Range with unstable flow; long strangle owns potential breakout (L2).",
                20,
                45,
                0.40,
                0.02,
                long_sigma=0.75,
                hedge_sigma=-0.75,
                defined_risk=True,
            )
        if volatility == "high_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "long_strangle",
                regime_key,
                "High-vol range breakdown risk; long strangle owns the move (L2).",
                14,
                30,
                0.35,
                0.02,
                long_sigma=0.5,
                hedge_sigma=-0.5,
                defined_risk=True,
            )
        if volatility == "high_vol" and dealer_flow == "supportive":
            return _candidate(
                "long_straddle",
                regime_key,
                "Vol-rich range; long straddle for maximum convexity (L2).",
                14,
                30,
                0.35,
                0.02,
                long_sigma=0.0,
                hedge_sigma=0.0,
                defined_risk=True,
            )
        # Default range: long strangle
        return _candidate(
            "long_strangle",
            regime_key,
            "Default range defined-risk long strangle (L2).",
            20,
            45,
            0.35,
            0.015,
            long_sigma=0.75,
            hedge_sigma=-0.75,
            defined_risk=True,
        )

    # Transition direction states
    return _candidate(
        "no_trade_or_micro_position",
        regime_key,
        "Ambiguous direction; default to no trade or minimal defined-risk probe.",
        14,
        21,
        0.25,
        0.005,
        defined_risk=True,
    )
