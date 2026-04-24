"""
News Fetcher for RLM Universe.
Pulls news articles from providers like Finnhub.
"""

from __future__ import annotations

import datetime
import pandas as pd
from typing import Any

try:
    import finnhub
except ImportError:
    finnhub = None

class NewsFetcher:
    def __init__(self, api_key: str, source: str = "finnhub"):
        self.source = source
        self.api_key = api_key
        if source == "finnhub" and finnhub:
            self.client = finnhub.Client(api_key=api_key)
        else:
            self.client = None

    def fetch_for_universe(self, symbols: list[str], lookback_hours: int = 24) -> pd.DataFrame:
        """
        Fetch company news for each symbol in the universe for the last N hours.
        """
        if not self.client:
            return pd.DataFrame()

        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=lookback_hours)
        
        # Finnhub expects YYYY-MM-DD
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        all_articles = []
        for sym in symbols:
            try:
                # Finnhub company_news returns a list of dicts
                news = self.client.company_news(sym, _from=start_str, to=end_str)
                for article in news:
                    article["symbol"] = sym
                    all_articles.append(article)
            except Exception:
                continue

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        # Filter by timestamp to be precise with lookback_hours if needed
        # df["datetime"] is unix timestamp in finnhub
        if "datetime" in df.columns:
            threshold = start.timestamp()
            df = df[df["datetime"] >= threshold]
            
        return df
