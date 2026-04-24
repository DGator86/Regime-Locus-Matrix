"""
Morning Briefing Pipeline.
Enriches the standard RLM pipeline with morning-specific data (overnight, pre-market, news).
"""

from __future__ import annotations

import pandas as pd
from typing import Any
from rlm.core.pipeline import FullRLMPipeline, PipelineResult, FullRLMConfig

class MorningBriefingPipeline:
    def __init__(self, config: FullRLMConfig | None = None):
        self.config = config or FullRLMConfig()
        self.base_pipeline = FullRLMPipeline(self.config)

    def run(
        self,
        symbol: str,
        prior_bars: pd.DataFrame,
        prior_options: pd.DataFrame | None,
        overnight: pd.DataFrame,
        news: pd.DataFrame,
        premarket: pd.DataFrame
    ) -> PipelineResult:
        """
        Runs the full pipeline with enriched data.
        """
        # 1. Build Enriched DataFrame
        df = self._build_enriched_df(symbol, prior_bars, overnight, news, premarket)
        
        # 2. Run Base Pipeline
        return self.base_pipeline.run(df, option_chain_df=prior_options)

    def _build_enriched_df(
        self,
        symbol: str,
        bars: pd.DataFrame,
        overnight: pd.DataFrame,
        news: pd.DataFrame,
        premarket: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Joins morning data onto the bar DataFrame.
        """
        df = bars.copy()
        
        # Add Overnight Metrics (applied to the last bar or as new columns)
        if not overnight.empty:
            if "futures" in overnight.columns:
                # Add ES/NQ changes as global factors for this symbol
                for fut_sym, change in overnight["futures"].items():
                    df[f"overnight_future_{fut_sym.lower()}_change"] = change
            
            if "symbol_changes" in overnight.columns and symbol in overnight["symbol_changes"].index:
                df["overnight_symbol_change_pct"] = overnight["symbol_changes"].loc[symbol]

        # Add News Metrics
        if not news.empty and symbol in news.index:
            s = news.loc[symbol]
            df["news_avg_sentiment"] = s.get("avg_sentiment", 0.0)
            df["news_article_count"] = s.get("article_count", 0)
            # Tags could be encoded or joined differently, here we just count them
            df["news_tag_count"] = len(s.get("relevant_tags", []))

        # Add Pre-market Metrics
        if not premarket.empty:
            # premarket is expected to be OHLCV of the pre-market session
            # We can append it or use it to calculate gap/volume features
            # For now, let's just calculate a few summary stats
            df["premarket_volume"] = premarket["volume"].sum()
            df["premarket_vwap"] = (premarket["close"] * premarket["volume"]).sum() / premarket["volume"].sum() if premarket["volume"].sum() > 0 else df["close"].iloc[-1]
            df["premarket_gap_pct"] = (premarket["close"].iloc[-1] / bars["close"].iloc[-1]) - 1 if not premarket.empty else 0.0

        # Fill NAs for the new morning columns
        morning_cols = [c for c in df.columns if c.startswith(("overnight_", "news_", "premarket_"))]
        df[morning_cols] = df[morning_cols].fillna(0.0)

        return df
