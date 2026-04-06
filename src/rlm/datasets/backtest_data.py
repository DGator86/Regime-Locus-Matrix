"""Build backtest / walk-forward inputs: IBKR equity history + aligned option-chain table + window manifest.

Massive option **snapshots** are point-in-time (no historical as-of chain in our integration). For long
histories we synthesize a chain grid from each bar's close (valid schema for the engine); swap in real
chain history later if you source it elsewhere.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from rlm.backtest.walkforward import WalkForwardConfig
from rlm.data.bs_greeks import bs_greeks_row


def synthetic_bars_demo(end: pd.Timestamp | str | date, periods: int = 220) -> pd.DataFrame:
    """Deterministic demo OHLCV (+ factor extras) for tests and dry runs."""
    rng = np.random.default_rng(42)
    end = pd.Timestamp(end).normalize()
    idx = pd.date_range(end=end, periods=periods, freq="D")
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
) -> pd.DataFrame:
    """One chain snapshot per bar timestamp (calls + puts), strikes around spot.

    Includes BS greeks, open interest, and tight but non-zero spreads so enrichment and
    the backtest engine see realistic microstructure.
    """
    rows: list[dict[str, Any]] = []
    und = str(underlying).upper()
    rng = np.random.default_rng(42)
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
                rows.append(
                    {
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
                    }
                )
    return pd.DataFrame(rows)


def synthetic_option_chain_intraday_from_bars(
    bars: pd.DataFrame,
    underlying: str = "SPY",
    *,
    dte_days: int = 35,
    strike_offsets: tuple[float, ...] = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0),
    risk_free: float = 0.052,
    seed: int = 42,
) -> pd.DataFrame:
    """Intraday backtests: one greek grid per **session day**, replicated to each bar timestamp.

    Avoids recomputing BS on every 5m bar (very slow for multi-month histories) while preserving
    the exact ``timestamp`` rows the backtest engine matches bar-by-bar.
    """
    if bars.empty:
        return pd.DataFrame()

    und = str(underlying).upper()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    idx = bars.index
    day_key = pd.to_datetime(idx).normalize()

    for d in pd.Index(day_key.unique()).sort_values():
        sub = bars.loc[day_key == d]
        if sub.empty:
            continue
        spot = float(sub["close"].iloc[-1])
        ts0 = sub.index[0]
        expiry = pd.Timestamp(ts0) + pd.Timedelta(days=dte_days)
        dte = max((expiry.normalize() - pd.Timestamp(ts0).normalize()).days, 1)
        t_y = dte / 365.0
        iv = float(
            0.18 + 0.06 * abs(np.sin((float(pd.Timestamp(ts0).toordinal() % 400)) / 400.0 * np.pi))
        )

        day_templates: list[dict[str, Any]] = []
        for opt_type in ("call", "put"):
            is_call = opt_type == "call"
            for off in strike_offsets:
                strike = round((spot + off) / 5.0) * 5.0
                delta, gamma, _, vanna, charm = bs_greeks_row(
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
                day_templates.append(
                    {
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
                    }
                )

        for ts in sub.index:
            for tmpl in day_templates:
                r = dict(tmpl)
                r["timestamp"] = ts
                rows.append(r)

    return pd.DataFrame(rows)


def fetch_ibkr_daily_bars_range(
    symbol: str,
    start: pd.Timestamp | str | date,
    end: pd.Timestamp | str | date | None = None,
    *,
    chunk_days: int = 365,
    timeout_sec: float = 180.0,
) -> pd.DataFrame:
    """Pull daily bars from IBKR from ``start`` through ``end`` (inclusive) via chunked historical requests.

    Uses successive ``end_datetime`` anchors working backward until ``start`` is covered or IB returns
    no rows. Requires TWS/Gateway and ``pip install '.[ibkr]'``.
    """
    from rlm.data.ibkr_stocks import fetch_historical_stock_bars

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    parts: list[pd.DataFrame] = []
    cursor_end = end_ts
    iterations = 0
    max_iterations = 200

    while cursor_end >= start_ts and iterations < max_iterations:
        iterations += 1
        end_str = cursor_end.strftime("%Y%m%d 16:00:00 US/Eastern")
        duration = f"{chunk_days} D"
        chunk = fetch_historical_stock_bars(
            symbol,
            duration=duration,
            bar_size="1 day",
            what_to_show="TRADES",
            use_rth=1,
            end_datetime=end_str,
            timeout_sec=timeout_sec,
        )
        if chunk.empty:
            break
        chunk = chunk.copy()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        parts.append(chunk)
        earliest = pd.Timestamp(chunk["timestamp"].min()).normalize()
        if earliest <= start_ts:
            break
        cursor_end = earliest - timedelta(days=1)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.loc[(out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)]
    out = out.set_index("timestamp").sort_index()
    out.index.name = "timestamp"
    return out


def rolling_window_manifest(
    bars_index: pd.DatetimeIndex,
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """Rows describing each walk-forward fold (indices match :func:`~rlm.backtest.walkforward.run_walkforward`)."""
    n = len(bars_index)
    rows: list[dict[str, Any]] = []
    start = 0
    wid = 0
    while start + cfg.is_window + cfg.oos_window <= n:
        is_start = start
        is_end = start + cfg.is_window
        oos_end = is_end + cfg.oos_window
        rows.append(
            {
                "window_id": wid,
                "is_bar_start_idx": is_start,
                "is_bar_end_idx": is_end,
                "oos_bar_start_idx": is_end,
                "oos_bar_end_idx": oos_end,
                "is_start_date": bars_index[is_start],
                "is_end_date": bars_index[is_end - 1],
                "oos_start_date": bars_index[is_end],
                "oos_end_date": bars_index[oos_end - 1],
                "n_is_bars": cfg.is_window,
                "n_oos_bars": cfg.oos_window,
            }
        )
        wid += 1
        start += cfg.step_size
    return pd.DataFrame(rows)


def bars_to_csv_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars with DatetimeIndex -> CSV-ready frame with ``timestamp`` column."""
    out = bars.reset_index()
    if "timestamp" not in out.columns and out.columns[0] != "timestamp":
        out = out.rename(columns={out.columns[0]: "timestamp"})
    return out


def us_equity_rth_5m_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """5-minute bar open times Mon–Fri, 09:30–15:55 US/Eastern (RTH, no holiday calendar)."""
    start_d = pd.Timestamp(start).normalize()
    end_d = pd.Timestamp(end).normalize()
    if end_d < start_d:
        raise ValueError("end must be >= start")
    days = pd.bdate_range(start=start_d, end=end_d, freq="C")
    pieces: list[pd.DatetimeIndex] = []
    for d in days:
        day = pd.Timestamp(d).normalize()
        open_ = day + pd.Timedelta(hours=9, minutes=30)
        last_bar = day + pd.Timedelta(hours=15, minutes=55)
        pieces.append(pd.date_range(open_, last_bar, freq="5min"))
    if not pieces:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in pieces]))


def synthetic_5m_bars_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    end_price: float = 590.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic 5m OHLCV (+ ``vwap``) on :func:`us_equity_rth_5m_index` for intraday backtests."""
    idx = us_equity_rth_5m_index(start, end)
    if len(idx) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    n = len(idx)
    # Random walk in log space; scale for ~5m moves
    sigma = 0.00035
    log_p = np.log(end_price) + rng.normal(0, sigma, n).cumsum()
    log_p = log_p - log_p[-1] + np.log(float(end_price))
    close = np.exp(log_p)
    noise_h = np.abs(rng.normal(0, sigma * close, n)) * close
    noise_l = np.abs(rng.normal(0, sigma * close, n)) * close
    high = np.maximum(close + noise_h, close)
    low = np.minimum(close - noise_l, close)
    open_ = np.r_[close[0], close[:-1]]
    vol = (1_000_000 + (np.arange(n) % 40) * 25_000 + rng.integers(-50_000, 50_000, n)).clip(100_000)
    vwap = (high + low + close) / 3.0

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol.astype(float),
            "vwap": vwap,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def fetch_ibkr_5m_bars_range(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    chunk_days: int = 20,
    timeout_sec: float = 180.0,
) -> pd.DataFrame:
    """Pull **5-minute RTH** bars from IBKR from ``start`` through ``end`` (chunked requests).

    Requires TWS/Gateway and ``pip install '.[ibkr]'``. IBKR limits how much intraday history
    each request returns; we walk backward in ``chunk_days`` windows and merge.
    """
    from rlm.data.ibkr_stocks import fetch_historical_stock_bars

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    start_ts = start_ts.normalize()

    end_ts = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)

    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    parts: list[pd.DataFrame] = []
    cursor_end = end_ts + pd.Timedelta(days=1)
    iterations = 0
    max_iterations = 400
    duration = f"{int(chunk_days)} D"

    while cursor_end > start_ts and iterations < max_iterations:
        iterations += 1
        end_str = cursor_end.strftime("%Y%m%d 16:00:00 US/Eastern")
        chunk = fetch_historical_stock_bars(
            symbol,
            duration=duration,
            bar_size="5 mins",
            what_to_show="TRADES",
            use_rth=1,
            end_datetime=end_str,
            timeout_sec=timeout_sec,
        )
        if chunk.empty:
            break
        chunk = chunk.copy()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        parts.append(chunk)
        earliest = pd.Timestamp(chunk["timestamp"].min())
        if earliest <= start_ts:
            break
        cursor_end = earliest - pd.Timedelta(minutes=5)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.loc[(out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts + pd.Timedelta(days=1))]
    out = out.set_index("timestamp").sort_index()
    out.index.name = "timestamp"
    return out
