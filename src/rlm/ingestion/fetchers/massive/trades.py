"""Massive option trades fetcher."""

from __future__ import annotations

import pandas as pd

from rlm.data.massive import MassiveClient
from rlm.datasets.massive_paging import collect_massive_results


class MassiveOptionTradesFetcher:
    def __init__(self, client: MassiveClient | None = None) -> None:
        self.client = client or MassiveClient()

    def fetch(self, option_ticker: str, *, ts_gte: str, ts_lt: str, limit: int = 50_000) -> pd.DataFrame:
        first = self.client.option_trades(
            option_ticker,
            **{
                "timestamp.gte": ts_gte,
                "timestamp.lt": ts_lt,
                "limit": limit,
                "sort": "timestamp",
                "order": "asc",
            },
        )
        rows = collect_massive_results(self.client, first if isinstance(first, dict) else {})
        return pd.DataFrame(rows)
