"""Aggressive day-trader sniper filter for the $1K→$25K PDT Challenge.

Returns True only for A+ intraday setups that meet the full confluence gate:
VWAP position, relative volume, pre-market level break, and IV rank.

Regime label vocabulary matches the canonical RLM classifiers:
  direction  : "bull" | "bear" | "range" | "transition"
  vol        : "high_vol" | "low_vol" | "transition"
  liquidity  : "high_liquidity" | "low_liquidity"
  dealer_flow: "supportive" | "destabilizing"
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING

from rlm.data.paths import get_data_root

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# A+ setup gate
# ---------------------------------------------------------------------------


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
        Four-element tuple using canonical RLM labels:
        (direction, vol, liquidity, dealer_flow).
    current_bar:
        Latest 1-min OHLCV bar as a pandas Series with keys
        ``open``, ``high``, ``low``, ``close``, ``volume``.
    intraday_df:
        Full intraday 1-min OHLCV history for the current session.
    """
    try:
        direction, vol, liquidity, flow = regime
    except (TypeError, ValueError):
        return False

    if intraday_df is None or intraday_df.empty or current_bar is None:
        return False

    volume = _numeric_column(intraday_df, "volume")
    high = _numeric_column(intraday_df, "high")
    low = _numeric_column(intraday_df, "low")
    close = _numeric_column(intraday_df, "close")
    if volume is None or high is None or low is None or close is None:
        return False

    bar_volume = _numeric_bar_value(current_bar, "volume")
    bar_high = _numeric_bar_value(current_bar, "high")
    bar_low = _numeric_bar_value(current_bar, "low")
    bar_close = _numeric_bar_value(current_bar, "close")
    if bar_volume is None or bar_high is None or bar_low is None or bar_close is None:
        return False

    # Relative volume vs 20-bar rolling average
    avg_vol = volume.rolling(20).mean().iloc[-1] if len(intraday_df) >= 20 else 0.0
    rvol = bar_volume / avg_vol if math.isfinite(float(avg_vol)) and avg_vol > 0 else 0.0

    # Session VWAP
    volume_cumsum = volume.cumsum()
    if volume_cumsum.empty or not math.isfinite(float(volume_cumsum.iloc[-1])) or volume_cumsum.iloc[-1] <= 0:
        return False
    tp = (high + low + close) / 3.0
    vwap = (tp * volume).cumsum() / volume_cumsum
    current_vwap = float(vwap.iloc[-1])
    if not math.isfinite(current_vwap):
        return False

    # Pre-market levels (0.0 = bypass when no feed is wired)
    pm_high = get_premarket_high(symbol)
    pm_low = get_premarket_low(symbol)

    # IV rank (VIX-based for SPY/QQQ; CSV-based for others)
    iv_rank = get_iv_rank(symbol)

    # Bullish gate: dealer-supported bull + above VWAP + strong rvol + PM breakout + cheap premium
    if direction == "bull" and liquidity == "high_liquidity" and flow == "supportive":
        pm_break = pm_high == 0.0 or bar_high > pm_high
        if bar_close > current_vwap and rvol > 1.5 and pm_break and iv_rank < 40:
            return True

    # Bearish gate: bear momentum + below VWAP + strong rvol + PM breakdown
    if direction == "bear" and liquidity == "high_liquidity":
        pm_break = pm_low == 0.0 or bar_low < pm_low
        if bar_close < current_vwap and rvol > 1.5 and pm_break and iv_rank > 30:
            return True

    # Event straddle: high-vol regime + catalyst within 30 minutes
    if vol == "high_vol" and direction in ("bull", "bear"):
        if is_event_within_30min():
            return True

    return False


def _numeric_column(df: "pd.DataFrame", name: str) -> "pd.Series | None":
    """Return a finite numeric OHLCV column, accepting common case variants."""
    try:
        import pandas as pd  # noqa: PLC0415

        col_name = name if name in df.columns else None
        if col_name is None:
            for candidate in df.columns:
                if str(candidate).lower() == name:
                    col_name = candidate
                    break
        if col_name is None:
            return None
        series = pd.to_numeric(df[col_name], errors="coerce")
        return series if not series.empty else None
    except Exception:  # noqa: BLE001
        return None


def _numeric_bar_value(bar: object, name: str) -> float | None:
    """Read one numeric value from a Series/dict-like bar without raising."""
    value = None
    try:
        getter = getattr(bar, "get", None)
        if callable(getter):
            value = getter(name)
            if value is None:
                labels = getattr(bar, "index", None)
                if labels is None:
                    keys = getattr(bar, "keys", None)
                    labels = keys() if callable(keys) else ()
                for candidate in labels:
                    if str(candidate).lower() == name:
                        value = getter(candidate)
                        break
        elif name in bar:  # type: ignore[operator]
            value = bar[name]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# IV rank
# ---------------------------------------------------------------------------

# Date-keyed cache so the VIX download runs at most once per calendar day.
_vix_rank_cache: dict[str, float] = {}


def get_iv_rank(symbol: str, lookback_days: int = 252) -> float:
    """Return the min-max normalised IV rank (0–100) for ``symbol``.

    The rank is computed as::

        (current_iv - min_iv_in_window) / (max_iv_in_window - min_iv_in_window) * 100

    This is a min-max normalisation over the lookback window, *not* a
    strict percentile rank.  Thresholds (< 40 = cheap, > 30 = elevated)
    are calibrated to this scale.

    For SPY and QQQ the daily VIX close series is used as a real-time
    proxy requiring no options-chain history.  For other symbols the
    function reads ``<data_root>/iv_history/<SYMBOL>.csv`` (columns:
    ``date``, ``iv``) and returns 50 when the file is absent or thin.
    """
    if symbol.upper() in ("SPY", "QQQ"):
        return _vix_rank_cached()
    return _csv_iv_rank(symbol.upper(), lookback_days)


def _vix_rank_cached() -> float:
    """Return today's VIX min-max rank, downloading at most once per day."""
    today = datetime.date.today().isoformat()
    if today in _vix_rank_cache:
        return _vix_rank_cache[today]
    rank = _vix_rank_fresh()
    _vix_rank_cache[today] = rank
    return rank


def _vix_rank_fresh() -> float:
    """Download VIX history and compute min-max rank (0–100)."""
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
    filepath = get_data_root() / "iv_history" / f"{symbol}.csv"
    if not filepath.exists():
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
