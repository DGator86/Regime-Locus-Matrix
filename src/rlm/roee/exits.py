from __future__ import annotations


def should_exit_for_profit(
    pnl_pct: float,
    target_profit_pct: float,
) -> bool:
    return pnl_pct >= target_profit_pct


def should_exit_for_zone_breach(
    realized_price: float,
    lower_1s: float,
    upper_1s: float,
) -> bool:
    return (realized_price < lower_1s) or (realized_price > upper_1s)


def should_exit_for_regime_flip(
    entry_regime_key: str,
    current_regime_key: str,
) -> bool:
    return entry_regime_key != current_regime_key
