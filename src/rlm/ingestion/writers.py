"""High-level ingestion writers (fetch → normalize → write)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.lake import (
    option_ticker_file_slug,
    options_bars_1d_dir,
    options_bars_1m_dir,
    options_contracts_dir,
    options_quotes_dir,
    options_trades_dir,
    save_parquet,
    stock_1d_parquet,
    stock_1m_parquet,
)
from rlm.ingestion.fetchers.ibkr.stocks import IBKRStockFetcher
from rlm.ingestion.fetchers.massive.bars import MassiveOptionBarsFetcher
from rlm.ingestion.fetchers.massive.contracts import MassiveContractsFetcher
from rlm.ingestion.fetchers.massive.quotes import MassiveOptionQuotesFetcher
from rlm.ingestion.fetchers.massive.trades import MassiveOptionTradesFetcher


def write_ibkr_stock_parquet(
    symbol: str,
    *,
    duration: str,
    bar_size: str,
    duration_slug: str,
    out_path: Path | None = None,
    interval_dir: str = "1d",
    repo_root: Path | None = None,
    end_datetime: str = "",
    timeout_sec: float = 180.0,
) -> Path:
    df = IBKRStockFetcher().fetch_bars(
        symbol,
        duration=duration,
        bar_size=bar_size,
        end_datetime=end_datetime,
        timeout_sec=timeout_sec,
    )
    if out_path is None:
        out_path = (
            stock_1m_parquet(symbol, duration_slug=duration_slug, root=repo_root)
            if interval_dir == "1m"
            else stock_1d_parquet(symbol, duration_slug=duration_slug, root=repo_root)
        )
    save_parquet(df, Path(out_path))
    return Path(out_path)


def write_massive_option_contracts_parquet(
    underlying: str,
    *,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    **params,
) -> Path:
    df = MassiveContractsFetcher().fetch(underlying, **params)
    if out_path is None:
        slug = str(params.get("expiration_date") or "all").replace("-", "")
        out_path = options_contracts_dir(underlying, root=repo_root) / f"{underlying.lower()}_{slug}_contracts.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def write_massive_option_bars_parquet(
    option_ticker: str,
    *,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    underlying_for_path: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    **params,
) -> Path:
    df = MassiveOptionBarsFetcher().fetch(
        option_ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date,
        **params,
    )
    if not df.empty and "t" in df.columns:
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    if out_path is None:
        stem = option_ticker_file_slug(option_ticker)
        sub = options_bars_1d_dir(underlying_for_path, root=repo_root) if timespan == "day" else options_bars_1m_dir(underlying_for_path, root=repo_root)
        out_path = sub / f"{stem}_{from_date}_{to_date}_{timespan}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def write_massive_option_quotes_parquet(
    option_ticker: str,
    *,
    underlying_for_path: str,
    ts_gte: str,
    ts_lt: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    limit: int = 50_000,
) -> Path:
    df = MassiveOptionQuotesFetcher().fetch(option_ticker, ts_gte=ts_gte, ts_lt=ts_lt, limit=limit)
    if out_path is None:
        stem = option_ticker_file_slug(option_ticker)
        day_slug = ts_gte[:10].replace("-", "_")
        out_path = options_quotes_dir(underlying_for_path, root=repo_root) / f"{stem}_{day_slug}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)


def write_massive_option_trades_parquet(
    option_ticker: str,
    *,
    underlying_for_path: str,
    ts_gte: str,
    ts_lt: str,
    out_path: Path | None = None,
    repo_root: Path | None = None,
    limit: int = 50_000,
) -> Path:
    df = MassiveOptionTradesFetcher().fetch(option_ticker, ts_gte=ts_gte, ts_lt=ts_lt, limit=limit)
    if out_path is None:
        stem = option_ticker_file_slug(option_ticker)
        day_slug = ts_gte[:10].replace("-", "_")
        out_path = options_trades_dir(underlying_for_path, root=repo_root) / f"{stem}_{day_slug}.parquet"
    save_parquet(df, Path(out_path))
    return Path(out_path)
