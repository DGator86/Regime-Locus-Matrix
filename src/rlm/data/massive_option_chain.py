"""Map Massive **options snapshot** JSON to RLM :func:`~rlm.data.option_chain.normalize_option_chain` schema.

See `Option chain snapshot <https://massive.com/docs/rest/options/snapshots/option-chain-snapshot.md>`_.
Use :func:`collect_option_snapshot_pages` to follow ``next_url`` for full chains.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from rlm.data.massive import MassiveClient
from rlm.data.option_chain import normalize_option_chain
from rlm.data.occ_symbol import parse_occ_option_symbol


def collect_option_snapshot_pages(
    client: MassiveClient,
    underlying: str,
    **params: str | int | float | bool | None,
) -> list[dict[str, Any]]:
    """Fetch all pages of ``/v3/snapshot/options/{underlying}`` and concatenate ``results``."""
    data = client.option_chain_snapshot(underlying, **params)
    merged: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return merged
    merged.extend(data.get("results") or [])
    while isinstance(data, dict) and data.get("next_url"):
        data = client.get_by_url(str(data["next_url"]))
        if not isinstance(data, dict):
            break
        merged.extend(data.get("results") or [])
    return merged


def massive_option_snapshot_results_to_dataframe(
    results: list[dict[str, Any]],
    *,
    underlying: str,
    timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Convert snapshot ``results`` array to a frame ready for :func:`~rlm.data.option_chain.normalize_option_chain`."""
    und = str(underlying).upper()
    ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.utcnow().tz_localize(None)

    records: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        details = row.get("details")
        if not isinstance(details, dict):
            continue
        sym = details.get("ticker")
        if not sym:
            continue
        try:
            parsed = parse_occ_option_symbol(str(sym))
        except ValueError:
            continue

        lq = row.get("last_quote") if isinstance(row.get("last_quote"), dict) else {}
        bid = _float_or(lq.get("bid"))
        ask = _float_or(lq.get("ask"))

        # Fallback when last_quote is absent (Massive plan may not include real-time NBBO).
        # Synthesise bid/ask from the day close (last trade price) using a conservative spread.
        # 2 % of mid, floored at $0.02 per side — suitable for paper trading and chain matching.
        if bid is None or ask is None:
            day = row.get("day") if isinstance(row.get("day"), dict) else {}
            mid_price = _float_or(day.get("close")) or _float_or(day.get("vwap"))
            if mid_price is not None and mid_price > 0:
                half_spread = max(0.02, mid_price * 0.02) / 2.0
                bid = round(mid_price - half_spread, 2)
                ask = round(mid_price + half_spread, 2)

        exp = details.get("expiration_date")
        if exp:
            expiry = pd.Timestamp(str(exp))
        else:
            expiry = parsed.expiry

        ctype = str(details.get("contract_type", "")).lower().strip()
        if ctype not in ("call", "put"):
            ctype = parsed.option_type

        strike = details.get("strike_price")
        if strike is None:
            strike_f = parsed.strike
        else:
            try:
                strike_f = float(strike)
            except (TypeError, ValueError):
                strike_f = parsed.strike

        rec: dict[str, Any] = {
            "timestamp": ts,
            "underlying": und,
            "expiry": expiry,
            "option_type": ctype,
            "strike": strike_f,
            "bid": bid,
            "ask": ask,
            "contract_symbol": str(sym),
        }

        iv = row.get("implied_volatility")
        if iv is not None and iv != "":
            v = _float_or(iv)
            if v is not None:
                rec["iv"] = v

        greeks = row.get("greeks")
        if isinstance(greeks, dict):
            for k in ("delta", "gamma", "theta", "vega"):
                if greeks.get(k) is not None:
                    g = _float_or(greeks.get(k))
                    if g is not None:
                        rec[k] = g

        oi = row.get("open_interest")
        if oi is not None:
            try:
                rec["open_interest"] = int(oi)
            except (TypeError, ValueError):
                pass

        day = row.get("day") if isinstance(row.get("day"), dict) else {}
        vol = day.get("volume")
        if vol is not None:
            try:
                rec["volume"] = int(vol)
            except (TypeError, ValueError):
                pass

        records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "underlying",
                "expiry",
                "option_type",
                "strike",
                "bid",
                "ask",
            ]
        )

    return pd.DataFrame.from_records(records)


def _float_or(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def massive_option_snapshot_payload_to_dataframe(
    payload: Any,
    *,
    underlying: str,
    timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parse a single-page snapshot dict ``{ "results": [...] }``."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object with 'results'.")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("'results' must be a list.")
    return massive_option_snapshot_results_to_dataframe(rows, underlying=underlying, timestamp=timestamp)


def massive_option_snapshot_to_normalized_chain(
    payload: Any,
    *,
    underlying: str,
    timestamp: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parse one page of option snapshot JSON and return RLM-normalized chain."""
    raw = massive_option_snapshot_payload_to_dataframe(
        payload, underlying=underlying, timestamp=timestamp
    )
    if raw.empty:
        return raw
    return normalize_option_chain(raw)


def massive_option_chain_from_client(
    client: MassiveClient,
    underlying: str,
    *,
    timestamp: pd.Timestamp | str | None = None,
    **snapshot_params: str | int | float | bool | None,
) -> pd.DataFrame:
    """Fetch all snapshot pages and return a normalized option chain."""
    rows = collect_option_snapshot_pages(client, underlying, **snapshot_params)
    raw = massive_option_snapshot_results_to_dataframe(
        rows, underlying=underlying, timestamp=timestamp
    )
    if raw.empty:
        return raw
    return normalize_option_chain(raw)
