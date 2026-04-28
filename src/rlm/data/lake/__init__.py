"""Lake utilities and storage helpers."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .metadata import (
    lake_bars_path,
    lake_option_chain_path,
    lake_root,
    lake_symbol_dir,
)
from .writers import save_parquet


# Backward compatible path helpers used by ingestion code.
def data_lake_root(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path.cwd()
    return repo_root / "data"


def stock_1d_dir(symbol: str, *, root: Path | None = None) -> Path:
    return data_lake_root(root) / "stocks" / _safe_sym(symbol) / "1d"


def stock_1m_dir(symbol: str, *, root: Path | None = None) -> Path:
    return data_lake_root(root) / "stocks" / _safe_sym(symbol) / "1m"


def stock_1d_parquet(symbol: str, *, duration_slug: str, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return stock_1d_dir(sym, root=root) / f"{sym.lower()}_{duration_slug}_1d.parquet"


def stock_1m_parquet(symbol: str, *, duration_slug: str, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return stock_1m_dir(sym, root=root) / f"{sym.lower()}_{duration_slug}_1m.parquet"


def options_underlying_dir(symbol: str, *, root: Path | None = None) -> Path:
    return data_lake_root(root) / "options" / _safe_sym(symbol)


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


def options_flatfiles_dataset_dir(symbol: str, dataset: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "flatfiles" / str(dataset).strip().lower()


def options_flatfile_daily_parquet(
    symbol: str, dataset: str, trade_date: date, *, root: Path | None = None
) -> Path:
    return (
        options_flatfiles_dataset_dir(symbol, dataset, root=root)
        / f"{trade_date.isoformat()}.parquet"
    )


def option_ticker_file_slug(option_ticker: str) -> str:
    t = str(option_ticker).strip().upper().removeprefix("O:")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", t)


def _safe_sym(symbol: str) -> str:
    return str(symbol).strip().upper()


__all__ = [
    "lake_root",
    "lake_symbol_dir",
    "lake_bars_path",
    "lake_option_chain_path",
    "save_parquet",
    "data_lake_root",
    "stock_1d_dir",
    "stock_1m_dir",
    "stock_1d_parquet",
    "stock_1m_parquet",
    "options_underlying_dir",
    "options_contracts_dir",
    "options_bars_1d_dir",
    "options_bars_1m_dir",
    "options_trades_dir",
    "options_quotes_dir",
    "options_flatfiles_dataset_dir",
    "options_flatfile_daily_parquet",
    "option_ticker_file_slug",
]
