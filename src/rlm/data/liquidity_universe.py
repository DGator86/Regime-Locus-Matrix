"""Curated liquid US symbols for multi-ticker pulls (Massive, IBKR scans, etc.).

:data:`LIQUID_UNIVERSE` is the **Magnificent 7** plus **SPY** and **QQQ** (same names
typically among the most liquid, highest-capitalization US equities).

:data:`LIQUID_STOCK_UNIVERSE_10` keeps Mag7 + :data:`LIQUID_STOCK_EXTRAS` for workflows
that explicitly want a \"ten single-names\" list; it does **not** include the ETFs.

:data:`EXPANDED_LIQUID_UNIVERSE` is :data:`LIQUID_UNIVERSE` plus :data:`LIQUID_STOCK_EXTRAS`,
deduplicated (order-preserving). Used by ``rlm backtest --universe`` and walk-forward shims.
"""

from __future__ import annotations

MAGNIFICENT_SEVEN: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "MSFT",
    "NVDA",
    "TSLA",
)

# Three additional highly active single-names; replace to taste.
LIQUID_STOCK_EXTRAS: tuple[str, ...] = ("AMD", "AVGO", "JPM")

LIQUID_STOCK_UNIVERSE_10: tuple[str, ...] = MAGNIFICENT_SEVEN + LIQUID_STOCK_EXTRAS

CORE_LIQUID_ETFS: tuple[str, ...] = ("SPY", "QQQ")

# Mag7 + index ETFs — default universe for batch scripts and scans.
LIQUID_UNIVERSE: tuple[str, ...] = MAGNIFICENT_SEVEN + CORE_LIQUID_ETFS


def _dedupe_preserve_order(symbols: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return tuple(out)


# Liquid universe + stock-only extras (AMD, AVGO, JPM) with no duplicate tickers.
# Single source of truth for ``rlm backtest --universe``, walk-forward shims, and Spock.
EXPANDED_LIQUID_UNIVERSE: tuple[str, ...] = _dedupe_preserve_order(LIQUID_UNIVERSE + LIQUID_STOCK_EXTRAS)
