"""
News Tagger & Sentiment Analyzer.
Tags news articles with sentiment and relevance.
"""

from __future__ import annotations

import pandas as pd
from typing import Any
from itertools import chain

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

class NewsTagger:
    def __init__(self):
        if SentimentIntensityAnalyzer:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            
        # Placeholders for future FinBERT integration
        self.finbert = None

    def _analyze_sentiment(self, text: str) -> float:
        """
        Returns a sentiment score between -1.0 (very negative) and 1.0 (very positive).
        """
        if self.vader:
            scores = self.vader.polarity_scores(text)
            return float(scores["compound"])
        return 0.0

    def _extract_tags(self, text: str) -> list[str]:
        """
        Extract relevance tags based on keywords.
        """
        keywords = {
            "earnings": ["earnings", "eps", "report", "quarterly"],
            "guidance": ["guidance", "outlook", "forecast"],
            "ma": ["merger", "acquisition", "buyout", "takeover"],
            "macro": ["fed", "inflation", "cpi", "interest rates"],
            "fda": ["fda", "clinical trial", "phase"],
        }
        text_lower = text.lower()
        tags = []
        for tag, words in keywords.items():
            if any(word in text_lower for word in words):
                tags.append(tag)
        return tags

    def tag_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sentiment and relevance_tags columns to the DataFrame.
        """
        if df.empty:
            return df
            
        # Use 'headline' if available, otherwise 'summary'
        col = "headline" if "headline" in df.columns else "summary"
        if col not in df.columns:
            df["sentiment"] = 0.0
            df["relevance_tags"] = [[] for _ in range(len(df))]
            return df

        df["sentiment"] = df[col].apply(self._analyze_sentiment)
        df["relevance_tags"] = df[col].apply(self._extract_tags)
        return df

    def aggregate_per_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups by symbol and computes average sentiment, count, and unique tags.
        """
        if df.empty:
            return pd.DataFrame(columns=["avg_sentiment", "article_count", "relevant_tags"])

        agg = df.groupby("symbol").agg(
            avg_sentiment=("sentiment", "mean"),
            article_count=("symbol", "count"),
            relevant_tags=("relevance_tags", lambda x: list(set(chain(*x))))
        )
        return agg
