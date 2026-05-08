from __future__ import annotations

import math

from rlm.types.options import OptionLeg, TradeCandidate


def round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    return round(value / increment) * increment


def sigma_to_price_offset(
    current_price: float,
    sigma: float,
    sigma_multiple: float,
) -> float:
    return current_price * sigma * sigma_multiple


def target_strike_from_sigma(
    current_price: float,
    sigma: float,
    sigma_multiple: float,
    increment: float = 1.0,
) -> float:
    strike = current_price + sigma_to_price_offset(current_price, sigma, sigma_multiple)
    strike = round_to_increment(strike, increment)
    floor = max(float(increment), 1e-6)
    if not math.isfinite(strike) or strike < floor:
        strike = max(floor, round_to_increment(float(current_price), increment) or floor)
    return strike


def build_legs_from_candidate(
    candidate: TradeCandidate,
    current_price: float,
    sigma: float,
    strike_increment: float = 1.0,
) -> list[OptionLeg]:
    """
    Produces simplified option legs from sigma targets.
    This is not chain-aware matching yet.
    """
    name = candidate.strategy_name
    legs: list[OptionLeg] = []

    # ------------------------------------------------------------------
    # Level 2 single-leg strategies
    # ------------------------------------------------------------------
    if name == "long_call":
        long_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.5, strike_increment)
        legs = [OptionLeg(side="long", option_type="call", strike=long_strike)]

    elif name == "long_put":
        long_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or -0.5, strike_increment)
        legs = [OptionLeg(side="long", option_type="put", strike=long_strike)]

    # ------------------------------------------------------------------
    # Level 2 long straddle — buy ATM call + ATM put
    # ------------------------------------------------------------------
    elif name in {"long_straddle", "scalp_long_straddle"}:
        atm_call = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.0, strike_increment)
        atm_put = target_strike_from_sigma(current_price, sigma, candidate.hedge_sigma or 0.0, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="call", strike=atm_call),
            OptionLeg(side="long", option_type="put", strike=atm_put),
        ]

    # ------------------------------------------------------------------
    # Level 2 long debit spreads (replace diagonals / credit verticals)
    # ------------------------------------------------------------------
    elif name in {
        "long_call_spread",
        "bull_call_debit_spread",
        "small_bull_debit_spread",
        "bull_call_spread",
        "debit_spread_call",
        "0dte_bull_call_spread",
    }:
        long_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.5, strike_increment)
        short_strike = target_strike_from_sigma(current_price, sigma, candidate.short_sigma or 1.5, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="call", strike=long_strike),
            OptionLeg(side="short", option_type="call", strike=short_strike),
        ]

    elif name in {
        "long_put_spread",
        "bear_put_debit_spread",
        "small_bear_debit_spread",
        "bear_put_spread",
        "debit_spread_put",
        "0dte_bear_put_spread",
    }:
        long_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or -0.5, strike_increment)
        short_strike = target_strike_from_sigma(current_price, sigma, candidate.short_sigma or -1.5, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="put", strike=long_strike),
            OptionLeg(side="short", option_type="put", strike=short_strike),
        ]

    elif name == "long_strangle":
        call_strike = target_strike_from_sigma(current_price, sigma, abs(candidate.long_sigma or 0.5), strike_increment)
        put_strike = target_strike_from_sigma(
            current_price,
            sigma,
            -(abs(candidate.hedge_sigma or -0.5)),
            strike_increment,
        )
        legs = [
            OptionLeg(side="long", option_type="put", strike=put_strike),
            OptionLeg(side="long", option_type="call", strike=call_strike),
        ]

    elif name == "long_call_plus_put_hedge":
        call_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.5, strike_increment)
        put_strike = target_strike_from_sigma(current_price, sigma, candidate.hedge_sigma or -0.5, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="call", strike=call_strike),
            OptionLeg(side="long", option_type="put", strike=put_strike),
        ]

    elif name == "long_put_plus_call_hedge":
        put_strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or -0.5, strike_increment)
        call_strike = target_strike_from_sigma(current_price, sigma, candidate.hedge_sigma or 0.5, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="put", strike=put_strike),
            OptionLeg(side="long", option_type="call", strike=call_strike),
        ]

    elif name == "long_iron_condor":
        # Debit iron condor (Level 2): buy near-OTM put spread + near-OTM call spread.
        # Profits when underlying moves significantly in either direction (breakout structure).
        inner = candidate.wings_sigma_low or 1.0
        outer = candidate.wings_sigma_high or 1.5

        buy_put = target_strike_from_sigma(current_price, sigma, -inner, strike_increment)
        sell_put = target_strike_from_sigma(current_price, sigma, -outer, strike_increment)
        buy_call = target_strike_from_sigma(current_price, sigma, inner, strike_increment)
        sell_call = target_strike_from_sigma(current_price, sigma, outer, strike_increment)

        legs = [
            OptionLeg(side="long", option_type="put", strike=buy_put),
            OptionLeg(side="short", option_type="put", strike=sell_put),
            OptionLeg(side="long", option_type="call", strike=buy_call),
            OptionLeg(side="short", option_type="call", strike=sell_call),
        ]

    elif name in {
        "iron_condor",
        "small_iron_condor",
        "broken_wing_condor",
        "small_short_strangle_or_condor",
        "0dte_iron_condor",
        "1dte_iron_condor",
    }:
        # These names are retained for compatibility but all map to the same
        # debit iron condor leg structure now that we are Level 2 only.
        low = candidate.wings_sigma_low or 1.5
        high = candidate.wings_sigma_high or 2.0

        buy_put = target_strike_from_sigma(current_price, sigma, -low, strike_increment)
        sell_put = target_strike_from_sigma(current_price, sigma, -high, strike_increment)
        buy_call = target_strike_from_sigma(current_price, sigma, low, strike_increment)
        sell_call = target_strike_from_sigma(current_price, sigma, high, strike_increment)

        legs = [
            OptionLeg(side="long", option_type="put", strike=buy_put),
            OptionLeg(side="short", option_type="put", strike=sell_put),
            OptionLeg(side="long", option_type="call", strike=buy_call),
            OptionLeg(side="short", option_type="call", strike=sell_call),
        ]
    elif name == "calendar_spread":
        atm = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.0, strike_increment)
        legs = [
            OptionLeg(side="short", option_type="call", strike=atm),
            OptionLeg(side="long", option_type="call", strike=atm),
        ]

    # ------------------------------------------------------------------
    # Aggressive day-trader sniper legs (challenge module)
    # ------------------------------------------------------------------
    elif name == "aggressive_daytrader_call":
        strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.2, strike_increment)
        legs = [OptionLeg(side="long", option_type="call", strike=strike)]

    elif name == "aggressive_daytrader_put":
        strike = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or -0.2, strike_increment)
        legs = [OptionLeg(side="long", option_type="put", strike=strike)]

    elif name == "aggressive_daytrader_0DTE_straddle":
        atm = target_strike_from_sigma(current_price, sigma, candidate.long_sigma or 0.0, strike_increment)
        legs = [
            OptionLeg(side="long", option_type="call", strike=atm),
            OptionLeg(side="long", option_type="put", strike=atm),
        ]

    else:
        legs = []

    return legs
