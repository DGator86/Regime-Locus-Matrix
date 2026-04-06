"""Massive options reference + aggs + trades + quotes → Parquet (data lake layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rlm.data.massive import MassiveClient
from rlm.datasets.data_lake import (
    option_ticker_file_slug,
    options_bars_1d_dir,
    options_bars_1m_dir,
    options_contracts_dir,
    options_quotes_dir,
    options_trades_dir,
)
from rlm.datasets.data_lake_io import save_parquet
from rlm.datasets.massive_paging import collect_massive_results


def fetch_option_contracts_to_parquet(
    client: MassiveClient,
    underlying: str,
    *,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    **params: str | int | float | bool | None,
) -> Path:
    q: dict[str, str | int | float | bool | None] = {
        "underlying_ticker": str(underlying).upper(),
        **params,
    }
    first = client.option_contracts_reference(**q)
    rows = collect_massive_results(client, first if isinstance(first, dict) else {})
    df = pd.DataFrame(rows)
    if out_path is None:
        slug = params.get("expiration_date") or "all"
        slug = str(slug).replace("-", "")
        out_path = options_contracts_dir(underlying, root=repo_root) / f"{underlying.lower()}_{slug}_contracts.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def _aggs_results_to_frame(data: Any) -> pd.DataFrame:
    if not data or not isinstance(data, dict):
        return pd.DataFrame()
    rows = data.get("results", [])
    df = pd.DataFrame(rows)
    if not df.empty and "t" in df.columns:
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


def fetch_option_aggs_to_parquet(
    client: MassiveClient,
    options_ticker: str,
    *,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    underlying_for_path: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    **params: str | int | float | bool | None,
) -> Path:
    raw = client.option_aggs_range(
        options_ticker,
        multiplier,
        timespan,
        from_date,
        to_date,
        **params,
    )
    df = _aggs_results_to_frame(raw)
    stem = option_ticker_file_slug(options_ticker)
    if out_path is None:
        sub = options_bars_1d_dir(underlying_for_path, root=repo_root) if timespan == "day" else options_bars_1m_dir(
            underlying_for_path, root=repo_root
        )
        out_path = sub / f"{stem}_{from_date}_{to_date}_{timespan}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def fetch_option_trades_to_parquet(
    client: MassiveClient,
    options_ticker: str,
    *,
    underlying_for_path: str,
    ts_gte: str,
    ts_lt: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    limit: int = 50_000,
) -> Path:
    first = client.option_trades(
        options_ticker,
        **{"timestamp.gte": ts_gte, "timestamp.lt": ts_lt, "limit": limit, "sort": "timestamp", "order": "asc"},
    )
    rows = collect_massive_results(client, first if isinstance(first, dict) else {})
    df = pd.DataFrame(rows)
    stem = option_ticker_file_slug(options_ticker)
    if out_path is None:
        day_slug = ts_gte[:10].replace("-", "_")
        out_path = options_trades_dir(underlying_for_path, root=repo_root) / f"{stem}_{day_slug}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def fetch_option_quotes_to_parquet(
    client: MassiveClient,
    options_ticker: str,
    *,
    underlying_for_path: str,
    ts_gte: str,
    ts_lt: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    limit: int = 50_000,
) -> Path:
    first = client.option_quotes(
        options_ticker,
        **{"timestamp.gte": ts_gte, "timestamp.lt": ts_lt, "limit": limit, "sort": "timestamp", "order": "asc"},
    )
    rows = collect_massive_results(client, first if isinstance(first, dict) else {})
    df = pd.DataFrame(rows)
    stem = option_ticker_file_slug(options_ticker)
    if out_path is None:
        day_slug = ts_gte[:10].replace("-", "_")
        out_path = options_quotes_dir(underlying_for_path, root=repo_root) / f"{stem}_{day_slug}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)
