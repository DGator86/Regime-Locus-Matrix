"""Map Massive **options snapshot** JSON to RLM
:func:`~rlm.data.option_chain.normalize_option_chain` schema.

See `Option chain snapshot <https://massive.com/docs/rest/options/snapshots/option-chain-snapshot.md>`_.
Use :func:`collect_option_snapshot_pages` to follow ``next_url`` for full chains.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Collection, Mapping

import pandas as pd

from rlm.data.massive import MassiveClient
from rlm.data.option_chain import normalize_option_chain
from rlm.data.occ_symbol import parse_occ_option_symbol

OptionSnapshotParam = str | int | float | bool | None
DEFAULT_HOT_CHAIN_CACHE_SYMBOLS = frozenset({"SPY", "QQQ"})


@dataclass
class MassiveOptionChainBatchResult:
    """Batch fetch output for multi-ticker option chain loads."""

    chains: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    cache_hits: set[str] = field(default_factory=set)


@dataclass
class _OptionChainCacheEntry:
    fetched_at_monotonic: float
    chain: pd.DataFrame


_OPTION_CHAIN_RAM_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], _OptionChainCacheEntry] = {}
_OPTION_CHAIN_RAM_CACHE_LOCK = Lock()


def collect_option_snapshot_pages(
    client: MassiveClient,
    underlying: str,
    **params: OptionSnapshotParam,
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
    """Convert snapshot ``results`` array to a frame ready for normalization."""
    und = str(underlying).upper()
    ts = (
        pd.Timestamp(timestamp)
        if timestamp is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )

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
    return massive_option_snapshot_results_to_dataframe(
        rows,
        underlying=underlying,
        timestamp=timestamp,
    )


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
    cache_ttl_s: float = 0.0,
    use_ram_cache: bool | None = None,
    hot_cache_symbols: Collection[str] | None = None,
    **snapshot_params: OptionSnapshotParam,
) -> pd.DataFrame:
    """Fetch all snapshot pages and return a normalized option chain."""
    chain, _ = _massive_option_chain_from_client_with_cache_state(
        client,
        underlying,
        timestamp=timestamp,
        cache_ttl_s=cache_ttl_s,
        use_ram_cache=use_ram_cache,
        hot_cache_symbols=hot_cache_symbols,
        **snapshot_params,
    )
    return chain


def massive_option_chains_from_client(
    client: MassiveClient,
    underlyings: Collection[str],
    *,
    timestamps: Mapping[str, pd.Timestamp | str | None] | None = None,
    per_symbol_params: Mapping[str, Mapping[str, OptionSnapshotParam]] | None = None,
    max_workers: int = 4,
    cache_ttl_s: float = 0.0,
    use_ram_cache: bool | None = None,
    hot_cache_symbols: Collection[str] | None = None,
    **snapshot_params: OptionSnapshotParam,
) -> MassiveOptionChainBatchResult:
    """Fetch multiple underlyings concurrently using the existing synchronous client."""
    symbols = _unique_symbols(underlyings)
    batch = MassiveOptionChainBatchResult()
    if not symbols:
        return batch

    def _load_one(sym: str) -> tuple[pd.DataFrame, bool]:
        params = dict(snapshot_params)
        if per_symbol_params and sym in per_symbol_params:
            params.update(per_symbol_params[sym])
        ts = timestamps.get(sym) if timestamps else None
        return _massive_option_chain_from_client_with_cache_state(
            client,
            sym,
            timestamp=ts,
            cache_ttl_s=cache_ttl_s,
            use_ram_cache=use_ram_cache,
            hot_cache_symbols=hot_cache_symbols,
            **params,
        )

    if max_workers <= 1 or len(symbols) == 1:
        for sym in symbols:
            try:
                chain, cache_hit = _load_one(sym)
            except Exception as exc:
                batch.errors[sym] = str(exc)
                continue
            batch.chains[sym] = chain
            if cache_hit:
                batch.cache_hits.add(sym)
        return batch

    worker_count = min(max(1, int(max_workers)), len(symbols))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="massive-chain",
    ) as executor:
        futures = {executor.submit(_load_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                chain, cache_hit = future.result()
            except Exception as exc:
                batch.errors[sym] = str(exc)
                continue
            batch.chains[sym] = chain
            if cache_hit:
                batch.cache_hits.add(sym)
    return batch


def clear_massive_option_chain_ram_cache() -> None:
    """Clear the in-process RAM cache used for hot chains."""
    with _OPTION_CHAIN_RAM_CACHE_LOCK:
        _OPTION_CHAIN_RAM_CACHE.clear()


def _massive_option_chain_from_client_with_cache_state(
    client: MassiveClient,
    underlying: str,
    *,
    timestamp: pd.Timestamp | str | None = None,
    cache_ttl_s: float = 0.0,
    use_ram_cache: bool | None = None,
    hot_cache_symbols: Collection[str] | None = None,
    **snapshot_params: OptionSnapshotParam,
) -> tuple[pd.DataFrame, bool]:
    symbol = str(underlying).upper().strip()
    cache_enabled = _should_use_ram_cache(
        symbol,
        cache_ttl_s=cache_ttl_s,
        use_ram_cache=use_ram_cache,
        hot_cache_symbols=hot_cache_symbols,
    )
    cache_key = (symbol, _freeze_snapshot_params(snapshot_params))
    if cache_enabled:
        cached = _get_cached_chain(cache_key, ttl_s=cache_ttl_s)
        if cached is not None:
            return cached, True

    rows = collect_option_snapshot_pages(client, underlying, **snapshot_params)
    raw = massive_option_snapshot_results_to_dataframe(
        rows, underlying=underlying, timestamp=timestamp
    )
    chain = raw if raw.empty else normalize_option_chain(raw)
    if cache_enabled:
        _store_cached_chain(cache_key, chain)
        return chain.copy(deep=False), False
    return chain, False


def _unique_symbols(underlyings: Collection[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for underlying in underlyings:
        sym = str(underlying).upper().strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _freeze_snapshot_params(
    params: Mapping[str, OptionSnapshotParam],
) -> tuple[tuple[str, str], ...]:
    frozen: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        frozen.append((str(key), str(value)))
    return tuple(sorted(frozen))


def _should_use_ram_cache(
    symbol: str,
    *,
    cache_ttl_s: float,
    use_ram_cache: bool | None,
    hot_cache_symbols: Collection[str] | None,
) -> bool:
    if cache_ttl_s <= 0:
        return False
    if use_ram_cache is not None:
        return bool(use_ram_cache)
    raw_hot_symbols = (
        DEFAULT_HOT_CHAIN_CACHE_SYMBOLS if hot_cache_symbols is None else hot_cache_symbols
    )
    hot = {
        str(sym).upper().strip()
        for sym in raw_hot_symbols
        if str(sym).strip()
    }
    return symbol in hot


def _get_cached_chain(
    cache_key: tuple[str, tuple[tuple[str, str], ...]],
    *,
    ttl_s: float,
) -> pd.DataFrame | None:
    now = monotonic()
    with _OPTION_CHAIN_RAM_CACHE_LOCK:
        cached = _OPTION_CHAIN_RAM_CACHE.get(cache_key)
        if cached is None:
            return None
        if now - cached.fetched_at_monotonic > ttl_s:
            _OPTION_CHAIN_RAM_CACHE.pop(cache_key, None)
            return None
        return cached.chain.copy(deep=False)


def _store_cached_chain(
    cache_key: tuple[str, tuple[tuple[str, str], ...]],
    chain: pd.DataFrame,
) -> None:
    with _OPTION_CHAIN_RAM_CACHE_LOCK:
        _OPTION_CHAIN_RAM_CACHE[cache_key] = _OptionChainCacheEntry(
            fetched_at_monotonic=monotonic(),
            chain=chain,
        )
