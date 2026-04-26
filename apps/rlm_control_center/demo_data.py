"""Synthetic OHLCV bars when no CSV / IBKR data is available."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_demo_bars(*, symbol: str = "SPY", n: int = 320) -> pd.DataFrame:
    """
    Create a deterministic synthetic OHLCV DataFrame indexed by business-day timestamps.
    
    Parameters:
        symbol (str): Ticker symbol for which data is generated (informational only).
        n (int): Number of business-day bars to generate.
    
    Returns:
        pd.DataFrame: DataFrame indexed by a DatetimeIndex named "timestamp" with columns
        "open", "high", "low", "close", and "volume".
    """
    rng = np.random.default_rng(42)
    # Over-generate by one week then slice to exactly n.  pd.bdate_range returns
    # fewer than `periods` elements when the end date falls on a weekend in newer
    # pandas versions; this pattern is always exact regardless of day-of-week.
    end = pd.Timestamp.now("UTC").normalize()
    idx = pd.bdate_range(end=end, periods=n + 7)[-n:]
    close = 450.0 + np.cumsum(rng.normal(0, 1.2, size=n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.35, size=n)
    high = np.maximum(open_, close) + rng.uniform(0.05, 1.1, size=n)
    low = np.minimum(open_, close) - rng.uniform(0.05, 1.1, size=n)
    vol = rng.integers(1_000_000, 50_000_000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        },
        index=idx,
    ).rename_axis("timestamp")


def get_demo_option_chain_stub(*, symbol: str = "SPY") -> pd.DataFrame:
    """
    Generate a minimal, synthetic option-chain table intended for SVI/IV demonstrations when no option-chain CSV is available.
    
    Parameters:
        symbol (str): Underlying symbol for which the synthetic option rows are produced (included for downstream compatibility).
    
    Returns:
        pd.DataFrame: Rows with columns:
            - `timestamp` (pd.Timestamp): UTC-normalized date for the generated chain.
            - `strike` (float): Option strike price.
            - `expiry` (str): Expiry date formatted as YYYY-MM-DD.
            - `dte` (float): Days to expiry.
            - `iv` (float): Implied volatility (clamped minimum 0.05).
            - `underlying_price` (float): Reference underlying price used to compute IV.
            - `volume` (float): Synthetic trade volume.
    """
    rng = np.random.default_rng(7)
    ts = pd.Timestamp.utcnow().normalize()
    strikes = np.arange(400.0, 520.0, 2.0)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for dte in (14.0, 30.0, 45.0):
        exp = (ts + pd.Timedelta(days=int(dte))).strftime("%Y-%m-%d")
        atm = 470.0
        for k in strikes:
            log_m = float(np.log(k / atm))
            iv = float(0.14 + 0.08 * log_m**2 + rng.normal(0, 0.008))
            iv = max(iv, 0.05)
            rows.append(
                {
                    "timestamp": ts,
                    "strike": float(k),
                    "expiry": exp,
                    "dte": dte,
                    "iv": iv,
                    "underlying_price": atm,
                    "volume": float(rng.integers(100, 5000)),
                }
            )
    return pd.DataFrame(rows)
