from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.lake.metadata import lake_bars_path, lake_option_chain_path


def lake_has_bars(
    symbol: str, data_root: str | Path | None = None, interval: str | None = None
) -> bool:
    return lake_bars_path(symbol, data_root, interval).is_file()


def lake_has_option_chain(symbol: str, data_root: str | Path | None = None) -> bool:
    return lake_option_chain_path(symbol, data_root).is_file()


def load_lake_bars(
    symbol: str, data_root: str | Path | None = None, interval: str | None = None
) -> pd.DataFrame:
    path = lake_bars_path(symbol, data_root, interval)
    if not path.is_file():
        raise FileNotFoundError(f"Lake bars not found for {symbol}: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp")
    else:
        df = df.sort_index()
    return df


def load_lake_option_chain(
    symbol: str,
    data_root: str | Path | None = None,
    as_of: str | None = None,
) -> pd.DataFrame | None:
    path = lake_option_chain_path(symbol, data_root, as_of)
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    for col in ("timestamp", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")
    return df
