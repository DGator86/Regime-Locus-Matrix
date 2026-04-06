"""Canonical on-disk paths: IBKR equities under ``data/stocks/``, Massive options under ``data/options/``."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def data_lake_root(repo_root: Path | None = None) -> Path:
    """Return ``<repo>/data`` (pass ``repo_root`` = repository root containing ``pyproject.toml``)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data"


def stock_1d_dir(symbol: str, *, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return data_lake_root(root) / "stocks" / sym / "1d"


def stock_1m_dir(symbol: str, *, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return data_lake_root(root) / "stocks" / sym / "1m"


def stock_1d_parquet(
    symbol: str,
    *,
    duration_slug: str,
    root: Path | None = None,
) -> Path:
    """e.g. ``data/stocks/SPY/1d/SPY_2y_1d.parquet``."""
    sym = _safe_sym(symbol)
    return stock_1d_dir(sym, root=root) / f"{sym.lower()}_{duration_slug}_1d.parquet"


def stock_1m_parquet(
    symbol: str,
    *,
    duration_slug: str,
    root: Path | None = None,
) -> Path:
    sym = _safe_sym(symbol)
    return stock_1m_dir(sym, root=root) / f"{sym.lower()}_{duration_slug}_1m.parquet"


def options_underlying_dir(symbol: str, *, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return data_lake_root(root) / "options" / sym


def options_contracts_dir(symbol: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "contracts"


def options_bars_1d_dir(symbol: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "bars_1d"


def options_bars_1m_dir(symbol: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "bars_1m"


def options_trades_dir(symbol: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "trades"


def options_quotes_dir(symbol: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "quotes"


def options_flatfiles_dataset_dir(
    symbol: str,
    dataset: str,
    *,
    root: Path | None = None,
) -> Path:
    """Massive Flat Files ingests: ``trades`` | ``quotes`` | ``day_aggs`` | ``minute_aggs``."""
    return options_underlying_dir(symbol, root=root) / "flatfiles" / str(dataset).strip().lower()


def options_flatfile_daily_parquet(
    symbol: str,
    dataset: str,
    trade_date: date,
    *,
    root: Path | None = None,
) -> Path:
    return options_flatfiles_dataset_dir(symbol, dataset, root=root) / f"{trade_date.isoformat()}.parquet"


def option_ticker_file_slug(option_ticker: str) -> str:
    """``O:SPY260619C00650000`` → safe filename stem."""
    t = str(option_ticker).strip().upper()
    t = t.removeprefix("O:")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", t)


def _safe_sym(symbol: str) -> str:
    return str(symbol).strip().upper()
