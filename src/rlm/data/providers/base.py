"""Provider contracts for ingestion market-data adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class ProviderBarsResult:
    bars_df: pd.DataFrame
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderOptionChainResult:
    chain_df: pd.DataFrame | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    def fetch_bars(
        self,
        *,
        symbol: str,
        start: str | None,
        end: str | None,
        interval: str,
    ) -> ProviderBarsResult: ...

    def fetch_option_chain(
        self,
        *,
        symbol: str,
    ) -> ProviderOptionChainResult: ...

