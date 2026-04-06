"""Curated liquid US symbols for multi-ticker pulls (Massive, IBKR scans, etc.).

:data:`LIQUID_UNIVERSE` is the **Magnificent 7** plus **SPY** and **QQQ** (same names
typically among the most liquid, highest-capitalization US equities).

:data:`LIQUID_STOCK_UNIVERSE_10` keeps Mag7 + :data:`LIQUID_STOCK_EXTRAS` for workflows
that explicitly want a \"ten single-names\" list; it does **not** include the ETFs.
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
