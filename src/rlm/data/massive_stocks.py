"""Massive **stock** aggregates and **order-flow proxies** from tick trades.

- OHLCV bars: :func:`massive_aggs_payload_to_bars_df` (from ``/v2/aggs/ticker/...``).
- Trades: :func:`massive_trades_payload_to_dataframe` (from ``/v3/trades/{ticker}``).
- Bar-level buy/sell size: :func:`trades_tick_rule_buy_sell` + :func:`aggregate_trade_flow_to_bars`
  (classic tick test; use with :class:`~rlm.factors.order_flow.OrderFlowFactors`).

Docs: `Stock trades <https://massive.com/docs/rest/stocks/trades-quotes/trades.md>`_,
`Stock quotes <https://massive.com/docs/rest/stocks/trades-quotes/quotes.md>`_,
`Custom bars <https://massive.com/docs/rest/stocks/aggregates/custom-bars.md>`_.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rlm.data.massive import MassiveClient


def collect_paged_results(
    client: MassiveClient, first_page: dict[str, Any]
) -> list[dict[str, Any]]:
    """Append all ``results`` from ``first_page`` and every ``next_url`` page."""
    if not isinstance(first_page, dict):
        return []
    merged: list[dict[str, Any]] = list(first_page.get("results") or [])
    data: dict[str, Any] | None = first_page
    while isinstance(data, dict) and data.get("next_url"):
        nxt = client.get_by_url(str(data["next_url"]))
        if not isinstance(nxt, dict):
            break
        merged.extend(nxt.get("results") or [])
        data = nxt
    return merged


def collect_stock_trades(
    client: MassiveClient,
    ticker: str,
    **params: str | int | float | bool | None,
) -> list[dict[str, Any]]:
    """Fetch all pages of ``GET /v3/trades/{ticker}``."""
    first = client.stock_trades(ticker, **params)
    if not isinstance(first, dict):
        return []
    return collect_paged_results(client, first)


def collect_stock_quotes(
    client: MassiveClient,
    ticker: str,
    **params: str | int | float | bool | None,
) -> list[dict[str, Any]]:
    """Fetch all pages of ``GET /v3/quotes/{ticker}``."""
    first = client.stock_quotes(ticker, **params)
    if not isinstance(first, dict):
        return []
    return collect_paged_results(client, first)


def massive_aggs_payload_to_bars_df(payload: Any) -> pd.DataFrame:
    """Build a bars frame from Massive aggregates JSON (``results`` with o,h,l,c,v,vw,t)."""
    if not isinstance(payload, dict):
        raise ValueError("Expected aggregates response dict.")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Missing 'results' list in aggregates response.")

    records: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = r.get("t")
        if t is None:
            continue
        try:
            ts = pd.Timestamp(int(t), unit="ms", tz="UTC").tz_convert(None)
        except (TypeError, ValueError, OSError):
            continue
        records.append(
            {
                "timestamp": ts,
                "open": float(r.get("o", np.nan)),
                "high": float(r.get("h", np.nan)),
                "low": float(r.get("l", np.nan)),
                "close": float(r.get("c", np.nan)),
                "volume": float(r.get("v", 0) or 0),
                "vwap": float(r["vw"]) if r.get("vw") is not None else np.nan,
            }
        )

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

    out = pd.DataFrame.from_records(records)
    return out.sort_values("timestamp").reset_index(drop=True)


def massive_trades_payload_to_dataframe(
    payload: Any,
    *,
    time_field: str = "sip_timestamp",
) -> pd.DataFrame:
    """Flatten ``/v3/trades/{ticker}`` ``results`` into a sorted DataFrame."""
    if not isinstance(payload, dict):
        raise ValueError("Expected trades response dict.")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Missing 'results' list.")

    recs: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ns = r.get(time_field)
        if ns is None:
            continue
        try:
            ns_i = int(ns)
        except (TypeError, ValueError):
            continue
        sz = r.get("size")
        try:
            size = float(sz) if sz is not None else 0.0
        except (TypeError, ValueError):
            size = 0.0
        recs.append(
            {
                time_field: ns_i,
                "price": float(r.get("price", np.nan)),
                "size": size,
                "exchange": r.get("exchange"),
                "conditions": r.get("conditions"),
            }
        )

    if not recs:
        return pd.DataFrame(columns=[time_field, "price", "size", "exchange", "conditions"])

    df = pd.DataFrame.from_records(recs)
    df = df.sort_values(time_field).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df[time_field], unit="ns", utc=True).dt.tz_convert(None)
    return df


def massive_quotes_payload_to_dataframe(
    payload: Any,
    *,
    time_field: str = "sip_timestamp",
) -> pd.DataFrame:
    """Flatten ``/v3/quotes/{ticker}`` ``results`` (NBBO history)."""
    if not isinstance(payload, dict):
        raise ValueError("Expected quotes response dict.")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Missing 'results' list.")

    recs: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ns = r.get(time_field)
        if ns is None:
            continue
        try:
            ns_i = int(ns)
        except (TypeError, ValueError):
            continue
        recs.append(
            {
                time_field: ns_i,
                "bid_price": float(r.get("bid_price", np.nan)),
                "bid_size": float(r.get("bid_size", 0) or 0),
                "ask_price": float(r.get("ask_price", np.nan)),
                "ask_size": float(r.get("ask_size", 0) or 0),
            }
        )

    if not recs:
        return pd.DataFrame(
            columns=[time_field, "bid_price", "bid_size", "ask_price", "ask_size", "timestamp"]
        )

    df = pd.DataFrame.from_records(recs)
    df = df.sort_values(time_field).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df[time_field], unit="ns", utc=True).dt.tz_convert(None)
    return df


def trades_tick_rule_buy_sell(trades: pd.DataFrame) -> pd.DataFrame:
    """Add ``side`` (+1 buy / -1 sell) via tick test on ``price`` (sorted by time)."""
    if trades.empty:
        out = trades.copy()
        out["side"] = pd.Series(dtype=np.int8)
        return out

    out = trades.sort_values("timestamp").reset_index(drop=True)
    prices = out["price"].to_numpy(dtype=float)
    n = len(prices)
    side = np.ones(n, dtype=np.int8)
    last = 1
    for i in range(n):
        if i == 0:
            side[i] = last
            continue
        if prices[i] > prices[i - 1]:
            last = 1
            side[i] = 1
        elif prices[i] < prices[i - 1]:
            last = -1
            side[i] = -1
        else:
            side[i] = last
    out["side"] = side
    return out


def aggregate_trade_flow_to_bars(
    bars: pd.DataFrame,
    trades_with_side: pd.DataFrame,
    *,
    bar_time_col: str = "timestamp",
    bar_duration: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Left-merge ``buy_volume`` and ``sell_volume`` onto bars (share count).

    Requires ``trades_with_side`` from :func:`trades_tick_rule_buy_sell` with
    ``timestamp``, ``size``, ``side``.
    """
    out = bars.copy()
    out[bar_time_col] = pd.to_datetime(out[bar_time_col])
    out = out.sort_values(bar_time_col).reset_index(drop=True)

    buy = np.zeros(len(out), dtype=float)
    sell = np.zeros(len(out), dtype=float)

    if trades_with_side.empty or "side" not in trades_with_side.columns:
        out["buy_volume"] = buy
        out["sell_volume"] = sell
        return out

    tdf = trades_with_side.dropna(subset=["timestamp", "size"]).copy()
    if tdf.empty:
        out["buy_volume"] = buy
        out["sell_volume"] = sell
        return out

    starts = out[bar_time_col].values.astype("datetime64[ns]")
    if bar_duration is None and len(starts) >= 2:
        d = out[bar_time_col].iloc[1] - out[bar_time_col].iloc[0]
        bar_duration = d if d > pd.Timedelta(0) else pd.Timedelta(minutes=1)
    elif bar_duration is None:
        bar_duration = pd.Timedelta(minutes=1)

    dur_ns = bar_duration.to_timedelta64()
    last_end = starts[-1] + dur_ns

    tt = tdf["timestamp"].values.astype("datetime64[ns]")
    sz = tdf["size"].to_numpy(dtype=float)
    sd = tdf["side"].to_numpy(dtype=np.int8)

    idx = np.searchsorted(starts, tt, side="right") - 1
    valid = (idx >= 0) & (tt >= starts[0]) & (tt < last_end)
    # assign to bar idx
    n_bars = len(buy)
    for i in np.flatnonzero(valid):
        j = int(idx[i])
        if j < 0 or j >= n_bars:
            continue
        if sd[i] >= 0:
            buy[j] += sz[i]
        else:
            sell[j] += sz[i]

    out["buy_volume"] = buy
    out["sell_volume"] = sell
    return out


def bars_with_flow_from_massive(
    aggs_payload: dict[str, Any],
    trades_list: list[dict[str, Any]],
    *,
    bar_time_col: str = "timestamp",
    bar_duration: pd.Timedelta | None = None,
    time_field: str = "sip_timestamp",
) -> pd.DataFrame:
    """Convenience: aggs JSON + flat trade rows → bars with OHLCV, vwap, buy/sell volume."""
    bars = massive_aggs_payload_to_bars_df(aggs_payload)
    trades_df = massive_trades_payload_to_dataframe({"results": trades_list}, time_field=time_field)
    tagged = trades_tick_rule_buy_sell(trades_df)
    return aggregate_trade_flow_to_bars(
        bars, tagged, bar_time_col=bar_time_col, bar_duration=bar_duration
    )
