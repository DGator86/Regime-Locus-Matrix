"""IBKR-backed market data provider."""

from __future__ import annotations

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.providers.base import ProviderBarsResult, ProviderOptionChainResult


class IBKRProvider:
    source = "ibkr"

    def fetch_bars(
        self,
        *,
        symbol: str,
        start: str | None,
        end: str | None,
        interval: str,
    ) -> ProviderBarsResult:
        del start, end
        bars = fetch_historical_stock_bars(symbol, bar_size=interval)
        return ProviderBarsResult(
            bars_df=bars,
            source=self.source,
            metadata={"rows": len(bars), "interval": interval},
        )

    def fetch_option_chain(self, *, symbol: str) -> ProviderOptionChainResult:
        return ProviderOptionChainResult(
            chain_df=None,
            source=self.source,
            metadata={"rows": 0, "detail": f"IBKR chain ingestion is not configured for {symbol}"},
        )
