"""Provider adapter registry for ingestion services."""

from rlm.data.providers.base import (
    MarketDataProvider,
    ProviderBarsResult,
    ProviderOptionChainResult,
)
from rlm.data.providers.ibkr_provider import IBKRProvider
from rlm.data.providers.yfinance_provider import YFinanceProvider


def resolve_provider(name: str) -> MarketDataProvider:
    key = name.lower().strip()
    if key == "yfinance":
        return YFinanceProvider()
    if key == "ibkr":
        return IBKRProvider()
    raise ValueError(f"Unsupported provider: {name}")


__all__ = [
    "IBKRProvider",
    "MarketDataProvider",
    "ProviderBarsResult",
    "ProviderOptionChainResult",
    "YFinanceProvider",
    "resolve_provider",
]
