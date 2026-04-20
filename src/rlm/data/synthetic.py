"""Synthetic data generators for tests, demos, and dry runs.

These are the canonical location for fixture data generators.
The old ``rlm.datasets.backtest_data`` module is a compatibility shim;
new code should import from here.

Usage::

    from rlm.data.synthetic import synthetic_bars_demo, synthetic_option_chain_from_bars
    import pandas as pd

    bars = synthetic_bars_demo(end=pd.Timestamp("2024-12-31"), periods=220)
    chain = synthetic_option_chain_from_bars(bars)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def synthetic_bars_demo(
    end: pd.Timestamp | str | date,
    periods: int = 220,
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic OHLCV + factor-extras bars suitable for tests and demos.

    The output matches the schema expected by ``FactorPipeline``.  The RNG seed
    is fixed so results are reproducible across runs.

    Parameters
    ----------
    end:
        Last bar date.
    periods:
        Number of daily bars to generate.
    seed:
        NumPy random seed (default 42).
    """
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end).normalize()
    idx = pd.date_range(end=end_ts, periods=periods, freq="D")
    base = np.linspace(620.0, 655.0, periods) + rng.normal(0, 1.2, periods).cumsum() * 0.15
    base = base - base[-1] + 655.0

    df = pd.DataFrame(
        {
            "open": base - 1.0,
            "high": base + 3.0,
            "low": base - 3.0,
            "close": base + np.sin(np.arange(periods) / 8.0) * 2.0,
            "volume": 1_500_000 + (np.arange(periods) % 10) * 50_000,
            "vwap": base,
            "anchored_vwap": base - 1.5,
            "buy_volume": 800_000 + (np.arange(periods) % 7) * 20_000,
            "sell_volume": 700_000 + (np.arange(periods) % 5) * 15_000,
            "advancers": 1800 + (np.arange(periods) % 20) * 10,
            "decliners": 1400 + (np.arange(periods) % 15) * 10,
            "index_return": pd.Series(base).pct_change(10).fillna(0.0).values,
            "vix": 16 + (np.arange(periods) % 12) * 0.4,
            "vvix": 88 + (np.arange(periods) % 9) * 1.2,
            "bid_ask_spread": 0.04 + (np.arange(periods) % 6) * 0.01,
            "order_book_depth": 5000 + (np.arange(periods) % 8) * 200,
            "gex": np.sin(np.arange(periods) / 10.0),
            "vanna": np.cos(np.arange(periods) / 12.0),
            "charm": np.sin(np.arange(periods) / 7.0),
            "put_call_skew": 0.02 + (np.arange(periods) % 5) * 0.004,
            "iv_rank": 0.45 + (np.arange(periods) % 10) * 0.02,
            "term_structure_ratio": 0.95 + (np.arange(periods) % 6) * 0.02,
            "dealer_position_proxy": np.sin(np.arange(periods) / 15.0) * 0.2,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def synthetic_option_chain_from_bars(
    bars: pd.DataFrame,
    underlying: str = "SPY",
    *,
    dte_days: int = 35,
    strike_offsets: tuple[float, ...] = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0),
    risk_free: float = 0.052,
    seed: int = 42,
) -> pd.DataFrame:
    """One chain snapshot per bar timestamp (calls + puts), strikes around spot.

    Includes Black-Scholes greeks, open interest, and realistic spreads so
    enrichment and the backtest engine see valid microstructure.

    Parameters
    ----------
    bars:
        OHLCV bars from ``synthetic_bars_demo`` or real data.
    underlying:
        Symbol string embedded in the output rows.
    dte_days:
        Days-to-expiry for synthesised contracts.
    strike_offsets:
        Dollar offsets from spot for strike selection.
    risk_free:
        Risk-free rate used in BS calculation.
    seed:
        NumPy random seed.
    """
    from rlm.data.bs_greeks import bs_greeks_row

    rows: list[dict[str, Any]] = []
    und = str(underlying).upper()
    rng = np.random.default_rng(seed)

    for ts, row in bars.iterrows():
        spot = float(row["close"])
        expiry = pd.Timestamp(ts) + pd.Timedelta(days=dte_days)
        dte = max((expiry.normalize() - pd.Timestamp(ts).normalize()).days, 1)
        t_y = dte / 365.0
        iv = float(
            0.18 + 0.06 * abs(np.sin((float(pd.Timestamp(ts).toordinal() % 400)) / 400.0 * np.pi))
        )
        for opt_type in ("call", "put"):
            is_call = opt_type == "call"
            for off in strike_offsets:
                strike = round((spot + off) / 5.0) * 5.0
                delta, gamma, vega, vanna, charm = bs_greeks_row(
                    spot=spot,
                    strike=strike,
                    time_years=t_y,
                    iv=iv,
                    risk_free=risk_free,
                    is_call=is_call,
                )
                mid = max(0.25, abs(strike - spot) * 0.06 + 0.8)
                spr = max(0.02, min(mid * 0.04, 1.5))
                oi = float(rng.integers(800, 25_000))
                vol = float(rng.integers(10, 5_000))
                rows.append({
                    "timestamp": ts,
                    "underlying": und,
                    "expiry": expiry,
                    "option_type": opt_type,
                    "strike": strike,
                    "bid": max(0.01, mid - spr / 2),
                    "ask": mid + spr / 2,
                    "delta": delta if np.isfinite(delta) else (0.45 if is_call else -0.45),
                    "gamma": gamma if np.isfinite(gamma) else np.nan,
                    "iv": iv,
                    "vanna": vanna if np.isfinite(vanna) else 0.0,
                    "charm": charm if np.isfinite(charm) else 0.0,
                    "open_interest": oi,
                    "volume": vol,
                })
    return pd.DataFrame(rows)
