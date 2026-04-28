"""Expiry and settlement logic for the backtesting engine.

This module handles:
- Intrinsic value calculation at expiration
- Structure-aware payoff resolution (verticals, condors, butterflies, strangles)
- Assignment/exercise simulation for short ITM legs
- Cash impact of settlement
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettlementResult:
    """Result of settling one or more option legs at expiry."""

    intrinsic_value: float
    """Net intrinsic value of the position per-unit (before multiplier)."""

    cash_impact: float
    """Cash credited (+) or debited (-) to the portfolio from settlement."""

    assignment_occurred: bool
    """True if any short ITM leg triggered assignment."""

    notes: str
    """Human-readable description of what happened."""


def _call_intrinsic(strike: float, underlying_price: float) -> float:
    return max(underlying_price - strike, 0.0)


def _put_intrinsic(strike: float, underlying_price: float) -> float:
    return max(strike - underlying_price, 0.0)


def leg_intrinsic_value(
    *,
    side: str,
    option_type: str,
    strike: float,
    underlying_price: float,
) -> float:
    """Return the signed intrinsic value contribution of a single leg.

    Long legs contribute positive intrinsic value; short legs contribute
    negative (the portfolio owes the exercise payout to the counterparty).
    """
    if option_type == "call":
        raw = _call_intrinsic(strike, underlying_price)
    elif option_type == "put":
        raw = _put_intrinsic(strike, underlying_price)
    else:
        raw = 0.0

    sign = 1.0 if side == "long" else -1.0
    return sign * raw


def settle_legs_at_expiry(
    *,
    legs: list[dict],
    underlying_price: float,
    contract_multiplier: int = 100,
) -> SettlementResult:
    """Compute settlement for a list of option legs at expiry.

    Each leg dict must contain: ``side``, ``option_type``, ``strike``.
    Quantity scaling is handled by the caller (portfolio layer).

    Returns a :class:`SettlementResult` with the net intrinsic value
    per-unit and the corresponding cash impact (before per-contract
    quantity scaling — that is applied in the portfolio layer).
    """
    total_intrinsic = 0.0
    assignment_occurred = False
    notes_parts: list[str] = []

    for leg in legs:
        side = str(leg.get("side", "long"))
        option_type = str(leg.get("option_type", "call"))
        strike = float(leg.get("strike", 0.0))

        intrinsic = leg_intrinsic_value(
            side=side,
            option_type=option_type,
            strike=strike,
            underlying_price=underlying_price,
        )
        total_intrinsic += intrinsic

        itm = intrinsic != 0.0
        if itm and side == "short":
            assignment_occurred = True
            notes_parts.append(
                f"short {option_type} K={strike} assigned (intrinsic={intrinsic:.4f})"
            )
        elif itm and side == "long":
            notes_parts.append(
                f"long {option_type} K={strike} exercised (intrinsic={intrinsic:.4f})"
            )
        else:
            notes_parts.append(f"{side} {option_type} K={strike} expired worthless")

    cash_impact = total_intrinsic * contract_multiplier

    return SettlementResult(
        intrinsic_value=total_intrinsic,
        cash_impact=cash_impact,
        assignment_occurred=assignment_occurred,
        notes="; ".join(notes_parts) if notes_parts else "all legs expired worthless",
    )


# ---------------------------------------------------------------------------
# Convenience helpers for common structures
# ---------------------------------------------------------------------------


def settle_vertical_spread(
    *,
    long_strike: float,
    short_strike: float,
    option_type: str,
    underlying_price: float,
    contract_multiplier: int = 100,
) -> SettlementResult:
    """Settle a vertical spread (debit or credit) at expiry.

    The long/short designation here refers to which strike is long and
    which is short, irrespective of whether it is a debit or credit spread.
    """
    legs = [
        {"side": "long", "option_type": option_type, "strike": long_strike},
        {"side": "short", "option_type": option_type, "strike": short_strike},
    ]
    return settle_legs_at_expiry(
        legs=legs,
        underlying_price=underlying_price,
        contract_multiplier=contract_multiplier,
    )


def settle_iron_condor(
    *,
    long_put_strike: float,
    short_put_strike: float,
    short_call_strike: float,
    long_call_strike: float,
    underlying_price: float,
    contract_multiplier: int = 100,
) -> SettlementResult:
    """Settle an iron condor at expiry."""
    legs = [
        {"side": "long", "option_type": "put", "strike": long_put_strike},
        {"side": "short", "option_type": "put", "strike": short_put_strike},
        {"side": "short", "option_type": "call", "strike": short_call_strike},
        {"side": "long", "option_type": "call", "strike": long_call_strike},
    ]
    return settle_legs_at_expiry(
        legs=legs,
        underlying_price=underlying_price,
        contract_multiplier=contract_multiplier,
    )


def settle_butterfly(
    *,
    lower_strike: float,
    middle_strike: float,
    upper_strike: float,
    option_type: str,
    underlying_price: float,
    contract_multiplier: int = 100,
) -> SettlementResult:
    """Settle a long butterfly spread at expiry.

    A standard long butterfly is: long 1 lower, short 2 middle, long 1 upper.
    """
    legs = [
        {"side": "long", "option_type": option_type, "strike": lower_strike},
        {"side": "short", "option_type": option_type, "strike": middle_strike},
        {"side": "short", "option_type": option_type, "strike": middle_strike},
        {"side": "long", "option_type": option_type, "strike": upper_strike},
    ]
    return settle_legs_at_expiry(
        legs=legs,
        underlying_price=underlying_price,
        contract_multiplier=contract_multiplier,
    )


def settle_strangle(
    *,
    put_strike: float,
    call_strike: float,
    side: str,
    underlying_price: float,
    contract_multiplier: int = 100,
) -> SettlementResult:
    """Settle a strangle (long or short) at expiry.

    ``side`` controls whether both legs are long (long strangle) or short
    (short strangle).
    """
    legs = [
        {"side": side, "option_type": "put", "strike": put_strike},
        {"side": side, "option_type": "call", "strike": call_strike},
    ]
    return settle_legs_at_expiry(
        legs=legs,
        underlying_price=underlying_price,
        contract_multiplier=contract_multiplier,
    )
