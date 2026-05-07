"""Aggressive day-trader sniper filter for the $1K→$25K PDT Challenge.

Returns True only for A+ intraday setups that meet the full confluence gate:
VWAP position, relative volume, pre-market level break, and IV rank.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def is_great_daytrade_setup(
    symbol: str,
    regime: tuple[str, str, str, str],
    current_bar: "pd.Series",
    intraday_df: "pd.DataFrame",
) -> bool:
    """Return True for A+ sniper setups only.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. "SPY", "QQQ").
    regime:
        Four-element tuple: (direction, vol, liquidity, dealer_flow).
    current_bar:
        Latest 1-min OHLCV bar as a pandas Series with keys
        ``open``, ``high``, ``low``, ``close``, ``volume``.
    intraday_df:
        Full intraday 1-min OHLCV history for the current session.
    """
    import pandas as pd  # noqa: PLC0415

    direction, vol, liquidity, flow = regime

    if intraday_df is None or intraday_df.empty or current_bar is None:
        return False

    # Relative volume vs 20-bar rolling average
    avg_vol = (
        intraday_df["volume"].rolling(20).mean().iloc[-1]
        if len(intraday_df) >= 20
        else 0.0
    )
    rvol = current_bar["volume"] / avg_vol if avg_vol > 0 else 0.0

    # Session VWAP
    tp = (intraday_df["high"] + intraday_df["low"] + intraday_df["close"]) / 3.0
    vwap = (tp * intraday_df["volume"]).cumsum() / intraday_df["volume"].cumsum()
    current_vwap = float(vwap.iloc[-1])

    # Pre-market levels (0.0 = bypass when no feed is wired)
    pm_high = get_premarket_high(symbol)
    pm_low = get_premarket_low(symbol)

    # IV rank (VIX percentile for SPY/QQQ; CSV-based for others)
    iv_rank = get_iv_rank(symbol)

    # Bullish gate: above VWAP + strong rvol + PM breakout + cheap premium
    if direction == "bullish" and flow == "buying_flow" and liquidity == "liquid":
        pm_break = pm_high == 0.0 or current_bar["high"] > pm_high
        if current_bar["close"] > current_vwap and rvol > 1.5 and pm_break and iv_rank < 40:
            return True

    # Bearish gate: below VWAP + strong rvol + PM breakdown + elevated IV
    if direction == "bearish" and flow == "selling_flow" and liquidity == "liquid":
        pm_break = pm_low == 0.0 or current_bar["low"] < pm_low
        if current_bar["close"] < current_vwap and rvol > 1.5 and pm_break and iv_rank > 30:
            return True

    # Event straddle: high-vol regime + catalyst within 30 minutes
    if vol == "high_vol" and direction in ("bullish", "bearish"):
        if is_event_within_30min():
            return True

    return False


# ---------------------------------------------------------------------------
# IV Rank
# ---------------------------------------------------------------------------


def get_iv_rank(symbol: str, lookback_days: int = 252) -> float:
    """Return IV rank (0–100) for ``symbol``.

    For SPY and QQQ the VIX percentile over the last year is used as a
    real-time substitute requiring no options-chain history.  For all other
    symbols the function reads ``data/iv_history/<SYMBOL>.csv``
    (columns: ``date``, ``iv``) and falls back to 50 when the file is absent
    or has fewer than 20 rows.
    """
    if symbol.upper() in ("SPY", "QQQ"):
        return _vix_percentile_rank()
    return _csv_iv_rank(symbol.upper(), lookback_days)


def _vix_percentile_rank() -> float:
    """Compute VIX percentile over the past year as an IV rank proxy."""
    try:
        import yfinance as yf  # noqa: PLC0415

        vix = yf.download("^VIX", period="1y", auto_adjust=False, progress=False)["Close"]
        if vix.empty or len(vix) < 20:
            return 50.0
        current = float(vix.iloc[-1])
        min_v = float(vix.min())
        max_v = float(vix.max())
        if max_v == min_v:
            return 50.0
        return round((current - min_v) / (max_v - min_v) * 100.0, 1)
    except Exception:  # noqa: BLE001
        return 50.0


def _csv_iv_rank(symbol: str, lookback_days: int) -> float:
    filepath = os.path.join("data", "iv_history", f"{symbol}.csv")
    if not os.path.exists(filepath):
        return 50.0
    try:
        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(filepath, parse_dates=["date"])
        if len(df) < 20:
            return 50.0
        current_iv = float(df["iv"].iloc[-1])
        recent = df.tail(lookback_days) if len(df) >= lookback_days else df
        min_iv = float(recent["iv"].min())
        max_iv = float(recent["iv"].max())
        if max_iv == min_iv:
            return 50.0
        return round((current_iv - min_iv) / (max_iv - min_iv) * 100.0, 1)
    except Exception:  # noqa: BLE001
        return 50.0


# ---------------------------------------------------------------------------
# Pre-market level stubs — replace with actual data feed
# ---------------------------------------------------------------------------


def get_premarket_high(symbol: str) -> float:
    """Return pre-market session high for ``symbol``.

    Returns 0.0 (bypass check) when no pre-market data is wired.
    Replace with a call to your minute-bar data feed or a shared state cache.
    """
    return 0.0


def get_premarket_low(symbol: str) -> float:
    """Return pre-market session low for ``symbol``.

    Returns 0.0 (bypass check) when no pre-market data is wired.
    Replace with a call to your minute-bar data feed or a shared state cache.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Event calendar stub — replace with actual calendar check
# ---------------------------------------------------------------------------


def is_event_within_30min() -> bool:
    """Return True if a scheduled macro event fires within the next 30 minutes.

    Replace with a call to your economic calendar feed
    (e.g. tradingeconomics, a local CSV of scheduled events, or earnings data).
    """
    return False
