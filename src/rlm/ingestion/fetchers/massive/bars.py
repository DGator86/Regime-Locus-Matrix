"""Massive option bars fetcher."""

from __future__ import annotations

import pandas as pd

from rlm.data.massive import MassiveClient


class MassiveOptionBarsFetcher:
    def __init__(self, client: MassiveClient | None = None) -> None:
        self.client = client or MassiveClient()

    def fetch(
        self,
        option_ticker: str,
        *,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        **params,
    ) -> pd.DataFrame:
        payload = self.client.option_aggs_range(
            option_ticker, multiplier, timespan, from_date, to_date, **params
        )
        if not isinstance(payload, dict):
            return pd.DataFrame()
        return pd.DataFrame(payload.get("results") or [])
