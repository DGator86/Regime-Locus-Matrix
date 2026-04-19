from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpreadStructure:
    strategy_name: str
    entry_price: float
    max_risk: float
    center_price: float
    width_abs: float
    long_strike: float | None = None
    short_strike: float | None = None
    lower_short: float | None = None
    upper_short: float | None = None


def build_bull_call_structure(start_close: float, width_abs: float) -> SpreadStructure:
    debit = max(0.20 * width_abs, 1e-3)
    return SpreadStructure(
        strategy_name="bull_call_spread",
        entry_price=debit,
        max_risk=debit,
        center_price=start_close,
        width_abs=width_abs,
        long_strike=start_close,
        short_strike=start_close + width_abs,
    )


def build_bear_put_structure(start_close: float, width_abs: float) -> SpreadStructure:
    debit = max(0.20 * width_abs, 1e-3)
    return SpreadStructure(
        strategy_name="bear_put_spread",
        entry_price=debit,
        max_risk=debit,
        center_price=start_close,
        width_abs=width_abs,
        long_strike=start_close,
        short_strike=start_close - width_abs,
    )


def build_iron_condor_structure(start_close: float, width_abs: float) -> SpreadStructure:
    credit = max(0.18 * width_abs, 1e-3)
    wing_width = max(0.5 * width_abs, 1e-3)
    return SpreadStructure(
        strategy_name="iron_condor",
        entry_price=credit,
        max_risk=wing_width - credit,
        center_price=start_close,
        width_abs=width_abs,
        lower_short=start_close - width_abs,
        upper_short=start_close + width_abs,
    )


def build_calendar_structure(start_close: float, width_abs: float) -> SpreadStructure:
    debit = max(0.16 * width_abs, 1e-3)
    return SpreadStructure(
        strategy_name="calendar_spread",
        entry_price=debit,
        max_risk=debit,
        center_price=start_close,
        width_abs=width_abs,
        long_strike=start_close,
        short_strike=start_close,
    )


def build_debit_spread_structure(
    start_close: float, width_abs: float, bias: float = 0.0
) -> SpreadStructure:
    debit = max(0.22 * width_abs, 1e-3)
    direction = 1.0 if bias >= 0 else -1.0
    return SpreadStructure(
        strategy_name="debit_spread",
        entry_price=debit,
        max_risk=debit,
        center_price=start_close,
        width_abs=width_abs,
        long_strike=start_close,
        short_strike=start_close + (direction * width_abs),
    )
