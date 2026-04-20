from __future__ import annotations

from pathlib import Path

from rlm.data.paths import get_data_root


def lake_root(data_root: str | Path | None = None) -> Path:
    return get_data_root(data_root) / "lake"


def lake_symbol_dir(symbol: str, data_root: str | Path | None = None) -> Path:
    return lake_root(data_root) / symbol.upper()


def lake_bars_path(symbol: str, data_root: str | Path | None = None, interval: str | None = None) -> Path:
    interval_slug = (interval or "1d").replace("/", "_")
    return lake_symbol_dir(symbol, data_root) / f"bars_{interval_slug}.parquet"


def lake_option_chain_path(symbol: str, data_root: str | Path | None = None, as_of: str | None = None) -> Path:
    suffix = f"_{as_of}" if as_of else ""
    return lake_symbol_dir(symbol, data_root) / f"option_chain{suffix}.parquet"
