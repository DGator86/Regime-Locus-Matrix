"""Massive contract fetcher."""

from __future__ import annotations

import pandas as pd

from rlm.data.massive import MassiveClient
from rlm.datasets.massive_paging import collect_massive_results


class MassiveContractsFetcher:
    def __init__(self, client: MassiveClient | None = None) -> None:
        self.client = client or MassiveClient()

    def fetch(self, underlying: str, **params) -> pd.DataFrame:
        first = self.client.option_contracts_reference(underlying_ticker=str(underlying).upper(), **params)
        rows = collect_massive_results(self.client, first if isinstance(first, dict) else {})
        return pd.DataFrame(rows)
