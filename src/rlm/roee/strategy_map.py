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


def get_strategy_for_regime(
    direction: str,
    volatility: str,
    liquidity: str,
    dealer_flow: str,
) -> TradeCandidate:
    regime_key = f"{direction}|{volatility}|{liquidity}|{dealer_flow}"

    # Bull states
    if direction == "bull":
        if (
            volatility == "low_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            return _candidate(
                "bull_call_debit_spread",
                regime_key,
                "Bullish directional regime with supportive flow and contained vol.",
                20,
                45,
                0.50,
                0.03,
                long_sigma=0.5,
                short_sigma=1.5,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "call_diagonal_spread",
                regime_key,
                "Bullish but flow less stable; prefer time-structure cushion.",
                30,
                60,
                0.40,
                0.025,
                long_sigma=0.5,
                short_sigma=1.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            return _candidate(
                "call_vertical_or_broken_wing_butterfly",
                regime_key,
                "Bullish expansion with enough liquidity; keep risk capped.",
                14,
                35,
                0.45,
                0.025,
                long_sigma=0.5,
                short_sigma=1.5,
                hedge_sigma=2.0,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "low_liquidity"
            and dealer_flow == "destabilizing"
        ):
            return _candidate(
                "long_call_plus_put_hedge",
                regime_key,
                "Bullish view but unstable and illiquid; convex protection needed.",
                14,
                30,
                0.35,
                0.02,
                long_sigma=0.5,
                hedge_sigma=-0.5,
                defined_risk=True,
            )
        return _candidate(
            "small_bull_debit_spread",
            regime_key,
            "Default bullish defined-risk participation.",
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
            return _candidate(
                "bear_put_debit_spread",
                regime_key,
                "Bearish directional regime with supportive downside flow.",
                20,
                45,
                0.50,
                0.03,
                long_sigma=-0.5,
                short_sigma=-1.5,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "put_diagonal_spread",
                regime_key,
                "Bearish but less orderly; use time-structure cushion.",
                30,
                60,
                0.40,
                0.025,
                long_sigma=-0.5,
                short_sigma=-1.5,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "high_liquidity"
            and dealer_flow == "supportive"
        ):
            return _candidate(
                "put_vertical_or_broken_wing_butterfly",
                regime_key,
                "Bearish expansion with adequate liquidity; defined risk.",
                14,
                35,
                0.45,
                0.025,
                long_sigma=-0.5,
                short_sigma=-1.5,
                hedge_sigma=-2.0,
                defined_risk=True,
            )
        if (
            volatility == "high_vol"
            and liquidity == "low_liquidity"
            and dealer_flow == "destabilizing"
        ):
            return _candidate(
                "long_put_plus_call_hedge",
                regime_key,
                "Bearish but unstable and illiquid; convex protection needed.",
                14,
                30,
                0.35,
                0.02,
                long_sigma=-0.5,
                hedge_sigma=0.5,
                defined_risk=True,
            )
        return _candidate(
            "small_bear_debit_spread",
            regime_key,
            "Default bearish defined-risk participation.",
            20,
            45,
            0.45,
            0.02,
            long_sigma=-0.5,
            short_sigma=-1.5,
            defined_risk=True,
        )

    # Range states
    if direction == "range":
        if volatility == "low_vol" and dealer_flow == "supportive":
            return _candidate(
                "iron_condor",
                regime_key,
                "Range with supportive flow and low vol; premium harvesting setup.",
                20,
                45,
                0.50,
                0.025,
                wings_sigma_low=1.5,
                wings_sigma_high=2.0,
                defined_risk=True,
            )
        if volatility == "low_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "broken_wing_condor",
                regime_key,
                "Range but unstable flow; bias protection on one side.",
                20,
                45,
                0.45,
                0.02,
                wings_sigma_low=1.5,
                wings_sigma_high=2.25,
                defined_risk=True,
            )
        if volatility == "high_vol" and dealer_flow == "destabilizing":
            return _candidate(
                "long_strangle",
                regime_key,
                "High-volatility range breakdown risk; own convexity.",
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
                "small_short_strangle_or_condor",
                regime_key,
                "Vol rich but must be tightly risk constrained.",
                14,
                30,
                0.35,
                0.015,
                wings_sigma_low=1.5,
                wings_sigma_high=2.0,
                defined_risk=True,
            )
        return _candidate(
            "small_iron_condor",
            regime_key,
            "Default range defined-risk premium structure.",
            20,
            45,
            0.40,
            0.015,
            wings_sigma_low=1.5,
            wings_sigma_high=2.0,
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
