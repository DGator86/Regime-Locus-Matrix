"""Storage-only data lake helpers (paths + parquet I/O)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd


def data_lake_root(repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data"


def stock_1d_dir(symbol: str, *, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return data_lake_root(root) / "stocks" / sym / "1d"


def stock_1m_dir(symbol: str, *, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return data_lake_root(root) / "stocks" / sym / "1m"


def stock_1d_parquet(symbol: str, *, duration_slug: str, root: Path | None = None) -> Path:
    sym = _safe_sym(symbol)
    return stock_1d_dir(sym, root=root) / f"{sym.lower()}_{duration_slug}_1d.parquet"


def stock_1m_parquet(symbol: str, *, duration_slug: str, root: Path | None = None) -> Path:
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


def options_flatfiles_dataset_dir(symbol: str, dataset: str, *, root: Path | None = None) -> Path:
    return options_underlying_dir(symbol, root=root) / "flatfiles" / str(dataset).strip().lower()


def options_flatfile_daily_parquet(symbol: str, dataset: str, trade_date: date, *, root: Path | None = None) -> Path:
    return options_flatfiles_dataset_dir(symbol, dataset, root=root) / f"{trade_date.isoformat()}.parquet"


def option_ticker_file_slug(option_ticker: str) -> str:
    t = str(option_ticker).strip().upper().removeprefix("O:")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", t)


def save_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=index)
    except ImportError as e:
        raise ImportError(
            "Parquet output requires pyarrow. Install: pip install pyarrow "
            "or pip install 'regime-locus-matrix[datalake]'"
        ) from e


def _safe_sym(symbol: str) -> str:
    return str(symbol).strip().upper()
