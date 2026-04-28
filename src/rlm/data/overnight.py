"""
Overnight & After-Hours Data Fetcher.
Fetches futures and equity moves outside of regular trading hours.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.utils.logging import get_logger

logger = get_logger(__name__)


def fetch_overnight_data(config: Any) -> pd.DataFrame:
    """
    Returns a DataFrame with key overnight metrics:
    - Futures change (ES, NQ) from prior close to current pre-market.
    - After-hours % change for symbols in universe.
    """
    logger.info("Fetching overnight data from IBKR...")

    # 1. Fetch futures (ES, NQ)
    # Note: These typically require specific symbols like 'ES' on 'CME'
    futures_symbols = ["ES", "NQ"]
    futures_data = {}

    for sym in futures_symbols:
        try:
            # Duration '1 D' with use_rth=0 to get the full 24h cycle
            df = fetch_historical_stock_bars(
                sym,
                duration="1 D",
                bar_size="1 hour",
                use_rth=0,
                exchange="CME" if sym in ["ES", "NQ"] else "SMART",
            )
            if not df.empty:
                # Net change from first bar (previous close / start of session) to last bar
                change = df["close"].iloc[-1] - df["close"].iloc[0]
                futures_data[sym] = change
        except Exception as e:
            logger.warning(f"Could not fetch overnight data for future {sym}: {e}")

    # 2. Fetch stock universe after-hours changes
    symbol_changes = {}
    universe = getattr(config, "universe", [])

    for sym in universe:
        try:
            # Fetch last 2 days of 1-hour bars with all hours
            df = fetch_historical_stock_bars(sym, duration="2 D", bar_size="1 hour", use_rth=0)
            if not df.empty and len(df) > 1:
                # Simplified: change from prior day's last RTH bar to current pre-market
                # For more precision, we'd find the 16:00 bar.
                # Here we'll just take current vs first bar in the 2-day set.
                change_pct = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
                symbol_changes[sym] = change_pct
        except Exception as e:
            logger.warning(f"Could not fetch after-hours data for {sym}: {e}")

    return pd.DataFrame(
        {"futures": pd.Series(futures_data), "symbol_changes": pd.Series(symbol_changes)}
    )
