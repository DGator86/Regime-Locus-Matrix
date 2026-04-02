from __future__ import annotations

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
    return round_to_increment(strike, increment)


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

    if name in {"bull_call_debit_spread", "small_bull_debit_spread"}:
        long_strike = target_strike_from_sigma(
            current_price, sigma, candidate.long_sigma or 0.5, strike_increment
        )
        short_strike = target_strike_from_sigma(
            current_price, sigma, candidate.short_sigma or 1.5, strike_increment
        )
        legs = [
            OptionLeg(side="long", option_type="call", strike=long_strike),
            OptionLeg(side="short", option_type="call", strike=short_strike),
        ]

    elif name in {"bear_put_debit_spread", "small_bear_debit_spread"}:
        long_strike = target_strike_from_sigma(
            current_price, sigma, candidate.long_sigma or -0.5, strike_increment
        )
        short_strike = target_strike_from_sigma(
            current_price, sigma, candidate.short_sigma or -1.5, strike_increment
        )
        legs = [
            OptionLeg(side="long", option_type="put", strike=long_strike),
            OptionLeg(side="short", option_type="put", strike=short_strike),
        ]

    elif name == "long_strangle":
        call_strike = target_strike_from_sigma(
            current_price, sigma, abs(candidate.long_sigma or 0.5), strike_increment
        )
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
        call_strike = target_strike_from_sigma(
            current_price, sigma, candidate.long_sigma or 0.5, strike_increment
        )
        put_strike = target_strike_from_sigma(
            current_price, sigma, candidate.hedge_sigma or -0.5, strike_increment
        )
        legs = [
            OptionLeg(side="long", option_type="call", strike=call_strike),
            OptionLeg(side="long", option_type="put", strike=put_strike),
        ]

    elif name == "long_put_plus_call_hedge":
        put_strike = target_strike_from_sigma(
            current_price, sigma, candidate.long_sigma or -0.5, strike_increment
        )
        call_strike = target_strike_from_sigma(
            current_price, sigma, candidate.hedge_sigma or 0.5, strike_increment
        )
        legs = [
            OptionLeg(side="long", option_type="put", strike=put_strike),
            OptionLeg(side="long", option_type="call", strike=call_strike),
        ]

    elif name in {
        "iron_condor",
        "small_iron_condor",
        "broken_wing_condor",
        "small_short_strangle_or_condor",
    }:
        low = candidate.wings_sigma_low or 1.5
        high = candidate.wings_sigma_high or 2.0

        short_put = target_strike_from_sigma(current_price, sigma, -low, strike_increment)
        long_put = target_strike_from_sigma(current_price, sigma, -high, strike_increment)
        short_call = target_strike_from_sigma(current_price, sigma, low, strike_increment)
        long_call = target_strike_from_sigma(current_price, sigma, high, strike_increment)

        legs = [
            OptionLeg(side="long", option_type="put", strike=long_put),
            OptionLeg(side="short", option_type="put", strike=short_put),
            OptionLeg(side="short", option_type="call", strike=short_call),
            OptionLeg(side="long", option_type="call", strike=long_call),
        ]

    elif name in {
        "call_diagonal_spread",
        "put_diagonal_spread",
        "call_vertical_or_broken_wing_butterfly",
        "put_vertical_or_broken_wing_butterfly",
    }:
        # simplified placeholder until expiry-aware chain matching is implemented
        sign_long = candidate.long_sigma or 0.5
        sign_short = candidate.short_sigma or 1.5
        opt_type = "call" if "call" in name else "put"
        long_strike = target_strike_from_sigma(
            current_price, sigma, sign_long, strike_increment
        )
        short_strike = target_strike_from_sigma(
            current_price, sigma, sign_short, strike_increment
        )
        legs = [
            OptionLeg(side="long", option_type=opt_type, strike=long_strike),
            OptionLeg(side="short", option_type=opt_type, strike=short_strike),
        ]

    else:
        legs = []

    return legs
