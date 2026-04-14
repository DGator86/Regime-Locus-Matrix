"""IBKR stock fetcher adapters."""

from __future__ import annotations

import pandas as pd

from rlm.data.ibkr_stocks import fetch_historical_stock_bars


class IBKRStockFetcher:
    def fetch_bars(
        self,
        symbol: str,
        *,
        duration: str,
        bar_size: str,
        end_datetime: str = "",
        timeout_sec: float = 180.0,
    ) -> pd.DataFrame:
        return fetch_historical_stock_bars(
            symbol,
            duration=duration,
            bar_size=bar_size,
            end_datetime=end_datetime,
            timeout_sec=timeout_sec,
        )
