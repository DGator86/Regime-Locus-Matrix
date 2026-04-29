"""Massive-backed market data provider."""

from __future__ import annotations

import datetime as dt

from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chain_from_client
from rlm.data.massive_stocks import massive_aggs_payload_to_bars_df
from rlm.data.providers.base import ProviderBarsResult, ProviderOptionChainResult


class MassiveProvider:
    source = "massive"

    def fetch_bars(
        self,
        *,
        symbol: str,
        start: str | None,
        end: str | None,
        interval: str,
    ) -> ProviderBarsResult:
        timespan = "day" if interval in {"1d", "1 day"} else "minute"
        multiplier = 1
        start_date = start or (dt.date.today() - dt.timedelta(days=60)).isoformat()
        end_date = end or dt.date.today().isoformat()

        client = MassiveClient()
        payload = client.stock_aggs_range(
            symbol,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
        )
        bars = massive_aggs_payload_to_bars_df(payload)
        if bars.empty:
            raise RuntimeError(f"massive returned no bars for {symbol}")
        return ProviderBarsResult(
            bars_df=bars,
            source=self.source,
            metadata={"rows": len(bars), "timespan": timespan, "multiplier": multiplier},
        )

    def fetch_option_chain(self, *, symbol: str) -> ProviderOptionChainResult:
        client = MassiveClient()
        chain = massive_option_chain_from_client(client, symbol)
        return ProviderOptionChainResult(
            chain_df=chain,
            source=self.source,
            metadata={"rows": 0 if chain is None else len(chain)},
        )
