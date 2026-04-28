#!/usr/bin/env python3
"""
Morning Briefing Scheduler.
Launch at 09:00 ET via cron.
Coordinates data collection, news ingestion, and 09:45 ET execution.
"""

import datetime
import time

from rlm.core.morning_pipeline import MorningBriefingPipeline
from rlm.core.pipeline import FullRLMConfig
from rlm.data.news.fetcher import NewsFetcher
from rlm.data.news.tagger import NewsTagger
from rlm.data.overnight import fetch_overnight_data
from rlm.data.paths import get_repo_root
from rlm.utils.logging import get_logger
from rlm.utils.market_hours import entry_window_open, session_label


# Mock/placeholder for pre-market monitor until full implementation
class PremarketMonitor:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.data = {}

    def get_final_data(self) -> dict[str, Any]:
        # Return empty dataframes for now
        import pandas as pd

        return {sym: pd.DataFrame() for sym in self.symbols}


def start_premarket_monitor(symbols: list[str]):
    return PremarketMonitor(symbols)


logger = get_logger(__name__)


def main():
    logger.info("Morning Briefing protocol initialized at %s", datetime.datetime.now())

    # 1. Load configuration
    root = get_repo_root()
    config_path = root / "configs" / "default.yaml"
    # Assuming FullRLMConfig.from_yaml exists or we can load manually
    # For now, we'll use a default config and populate from env if needed
    config = FullRLMConfig()
    # In a real scenario, we'd load universe from a shared config file
    universe = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "META", "MSFT", "AMZN", "GOOGL"]

    # 2. News collection & tagging (09:00 ET)
    news_api_key = ""  # Should come from config/env
    fetcher = NewsFetcher(api_key=news_api_key)
    tagger = NewsTagger()

    logger.info("Ingesting news for universe...")
    news_articles = fetcher.fetch_for_universe(universe, lookback_hours=24)
    tagged_news = tagger.tag_articles(news_articles)
    news_scores = tagger.aggregate_per_symbol(tagged_news)

    # 3. Fetch overnight / after-hours data
    logger.info("Fetching overnight price action...")
    overnight = fetch_overnight_data(config)

    # 4. Launch pre-market data stream
    logger.info("Starting pre-market monitoring...")
    premarket_stream = start_premarket_monitor(universe)

    # 5. Wait until 09:45 ET
    logger.info("Waiting for market open and 09:45 ET liquidity stabilization...")
    while not entry_window_open(buffer_open_minutes=15):
        current_state = session_label()
        logger.info(f"Market state: {current_state}. Waiting...")
        time.sleep(60)  # Poll every minute

    logger.info("09:45 ET reached. Finalizing directional thesis...")

    # Retrieve pre-market data
    premarket_data = premarket_stream.get_final_data()

    # 6. Run the Morning Briefing Pipeline
    pipeline = MorningBriefingPipeline(config)

    for sym in universe:
        logger.info(f"Processing {sym}...")
        try:
            # We need prior bars for each symbol
            # This is a simplification; in real use, we'd load these from disk/API
            from rlm.data.ibkr_stocks import fetch_historical_stock_bars

            prior_bars = fetch_historical_stock_bars(sym, duration="5 D", bar_size="1 day")

            result = pipeline.run(
                symbol=sym,
                prior_bars=prior_bars,
                prior_options=None,  # Optional
                overnight=overnight,
                news=news_scores,
                premarket=premarket_data.get(sym, pd.DataFrame()),
            )

            # Log results
            policy = result.policy_df.iloc[-1]
            logger.info(
                f"RESULT {sym}: Action={policy.get('roee_action')} Strategy={policy.get('roee_strategy')}"
            )

        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")

    logger.info("Morning Briefing protocol complete.")


if __name__ == "__main__":
    main()
