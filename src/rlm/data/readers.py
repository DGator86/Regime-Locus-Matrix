"""Central data readers with backend selection (auto/csv/lake)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.backend import DataBackend
from rlm.data.lake.readers import (
    lake_has_bars,
    lake_has_option_chain,
    load_lake_bars,
    load_lake_option_chain,
)
from rlm.data.paths import get_raw_data_dir
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def load_bars(
    symbol: str,
    bars_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str = "auto",
) -> pd.DataFrame:
    """Load OHLCV bars for *symbol*, returning a DataFrame indexed by timestamp.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    bars_path:
        Explicit path to a bars CSV.  When provided, *data_root* is ignored.
    data_root:
        Override the data root directory (see ``rlm.data.paths.get_data_root``).
        If omitted, ``RLM_DATA_ROOT`` / ``cwd/data`` is used.

    Raises
    ------
    FileNotFoundError
        When the resolved file does not exist.
    """
    path = _resolve_bars_path(symbol, bars_path, data_root, backend=backend)
    if not path.is_file():
        raise FileNotFoundError(
            f"Bars file not found: {path}\n"
            f"Fetch data first:  rlm ingest --symbol {symbol}\n"
            f"Or set a different root:  --data-root /your/path  or  RLM_DATA_ROOT=/your/path"
        )

    log.info("load_bars: %s (%s)", symbol, path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    interval: str | None = None,
) -> pd.DataFrame:
    selected = DataBackend.coerce(backend)
    if bars_path is not None:
        return _load_csv_bars(Path(bars_path).expanduser().resolve())

    if selected == DataBackend.LAKE:
        return load_lake_bars(symbol, data_root=data_root, interval=interval)

    if selected == DataBackend.AUTO and lake_has_bars(symbol, data_root=data_root, interval=interval):
        return load_lake_bars(symbol, data_root=data_root, interval=interval)

    csv_path = _resolve_bars_csv_path(symbol, data_root)
    if selected == DataBackend.AUTO or selected == DataBackend.CSV:
        return _load_csv_bars(csv_path)

    raise FileNotFoundError(f"No bars found for {symbol} using backend={selected.value}")


def load_option_chain(
    symbol: str,
    chain_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str = "auto",
) -> pd.DataFrame | None:
    """Load an option chain snapshot for *symbol*, or return ``None`` if absent.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    chain_path:
        Explicit path to a chain CSV.  When provided, *data_root* is ignored.
    data_root:
        Override the data root directory.
    """
    path = _resolve_chain_path(symbol, chain_path, data_root, backend=backend)
    if path is None or not path.is_file():
        log.debug("load_option_chain: no chain file for %s (looked at %s)", symbol, path)
        return None

    log.info("load_option_chain: %s (%s)", symbol, path)
    date_cols = [c for c in ["timestamp", "expiry"] if True]
    try:
        if path.suffix == ".parquet":
            out = pd.read_parquet(path)
            for c in date_cols:
                if c in out.columns:
                    out[c] = pd.to_datetime(out[c])
            return out
        return pd.read_csv(path, parse_dates=date_cols)
    except Exception as exc:
        log.warning("load_option_chain: failed to parse %s: %s", path, exc)
        return None
    as_of: str | None = None,
) -> pd.DataFrame | None:
    selected = DataBackend.coerce(backend)
    if chain_path is not None:
        return _load_csv_chain(Path(chain_path).expanduser().resolve())

    if selected == DataBackend.LAKE:
        df = load_lake_option_chain(symbol, data_root=data_root, as_of=as_of)
        if df is None:
            raise FileNotFoundError(f"Lake option chain not found for {symbol}")
        return df

    if selected == DataBackend.AUTO and lake_has_option_chain(symbol, data_root=data_root):
        return load_lake_option_chain(symbol, data_root=data_root, as_of=as_of)

    csv_path = _resolve_chain_csv_path(symbol, data_root)
    if selected in (DataBackend.AUTO, DataBackend.CSV):
        return _load_csv_chain(csv_path)

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
    return None


def _resolve_bars_csv_path(symbol: str, data_root: str | Path | None) -> Path:
    return get_raw_data_dir(data_root) / f"bars_{symbol.upper()}.csv"


def _resolve_chain_csv_path(symbol: str, data_root: str | Path | None) -> Path:
    return get_raw_data_dir(data_root) / f"option_chain_{symbol.upper()}.csv"


def _load_csv_bars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Bars file not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _load_csv_chain(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    for col in ("timestamp", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
