"""Central data readers with backend selection (auto/csv/lake)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rlm.data.backend import DataBackend
from rlm.data.lake.readers import lake_has_bars, lake_has_option_chain, load_lake_bars, load_lake_option_chain
from rlm.data.option_chain import option_chain_is_usable
from rlm.data.paths import get_raw_data_dir

_log = logging.getLogger(__name__)


def load_bars(
    symbol: str,
    bars_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str = "auto",
    interval: str | None = None,
) -> pd.DataFrame:
    selected = DataBackend.coerce(backend)
    if bars_path is not None:
        return _load_bars_file(Path(bars_path).expanduser().resolve())

    if selected == DataBackend.LAKE:
        return load_lake_bars(symbol, data_root=data_root, interval=interval)
    if selected == DataBackend.AUTO and lake_has_bars(symbol, data_root=data_root, interval=interval):
        return load_lake_bars(symbol, data_root=data_root, interval=interval)

    csv_path = _resolve_bars_csv_path(symbol, data_root)
    if selected in (DataBackend.AUTO, DataBackend.CSV):
        return _load_csv_bars(csv_path)

    raise FileNotFoundError(f"No bars found for {symbol} using backend={selected.value}")


def load_option_chain(
    symbol: str,
    chain_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str = "auto",
    as_of: str | None = None,
) -> pd.DataFrame | None:
    selected = DataBackend.coerce(backend)
    if chain_path is not None:
        return _load_chain_file(Path(chain_path).expanduser().resolve())

    if selected == DataBackend.LAKE:
        df = load_lake_option_chain(symbol, data_root=data_root, as_of=as_of)
        if df is None:
            raise FileNotFoundError(f"Lake option chain not found for {symbol}")
        return df
    if selected == DataBackend.AUTO and lake_has_option_chain(symbol, data_root=data_root):
        df_lake = load_lake_option_chain(symbol, data_root=data_root, as_of=as_of)
        if df_lake is not None:
            return df_lake

    if selected in (DataBackend.AUTO, DataBackend.CSV):
        return _load_csv_chain(_resolve_chain_csv_path(symbol, data_root))
    return None


def _resolve_bars_path(
    symbol: str,
    bars_path: str | Path | None,
    data_root: str | Path | None,
    *,
    backend: str,
) -> Path:
    if bars_path is not None:
        return Path(bars_path).expanduser().resolve()
    raw = get_raw_data_dir(data_root)
    sym = symbol.upper()
    parquet = raw / "lake" / "bars" / f"{sym}.parquet"
    csv = raw / f"bars_{sym}.csv"
    selected = backend.lower()
    if selected == "lake":
        return parquet
    if selected == "csv":
        return csv
    return parquet if parquet.exists() else csv


def _resolve_chain_path(
    symbol: str,
    chain_path: str | Path | None,
    data_root: str | Path | None,
    *,
    backend: str,
) -> Path | None:
    if chain_path is not None:
        return Path(chain_path).expanduser().resolve()
    raw = get_raw_data_dir(data_root)
    sym = symbol.upper()
    parquet = raw / "lake" / "chains" / f"{sym}.parquet"
    csv = raw / f"option_chain_{sym}.csv"
    selected = backend.lower()
    if selected == "lake":
        return parquet
    if selected == "csv":
        return csv
    return parquet if parquet.exists() else csv


def _resolve_bars_csv_path(symbol: str, data_root: str | Path | None) -> Path:
    return get_raw_data_dir(data_root) / f"bars_{symbol.upper()}.csv"


def _resolve_chain_csv_path(symbol: str, data_root: str | Path | None) -> Path:
    return get_raw_data_dir(data_root) / f"option_chain_{symbol.upper()}.csv"


def _load_bars_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return _load_parquet_bars(path)
    return _load_csv_bars(path)


def _load_chain_file(path: Path) -> pd.DataFrame | None:
    if path.suffix == ".parquet":
        return _load_parquet_chain(path)
    return _load_csv_chain(path)


def _load_parquet_bars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Bars file not found: {path}")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def _load_parquet_chain(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    for col in ("timestamp", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if not option_chain_is_usable(df):
        _log.warning(
            "Option chain %s missing required columns or empty; skipping chain enrichment",
            path,
        )
        return None
    return df


def _load_csv_bars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Bars file not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


def _load_csv_chain(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    for col in ("timestamp", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if not option_chain_is_usable(df):
        _log.warning(
            "Option chain %s missing required columns or empty; skipping chain enrichment",
            path,
        )
        return None
    return df
