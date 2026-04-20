"""Centralised data reader abstraction for RLM.

CLI commands and services must load bars and option chains through these
functions rather than calling ``pd.read_csv`` directly.  This keeps the
loading contract in one place and allows future evolution to Parquet/DuckDB
without touching CLI or service code.

Current backend: CSV files under the configured data root.
Future backend: Parquet lake / DuckDB — same interface, swapped internals.

Usage::

    from rlm.data.readers import load_bars, load_option_chain
    from rlm.data.paths import get_data_root

    bars = load_bars("SPY")                         # auto-resolves path
    bars = load_bars("SPY", bars_path="/my/file.csv")  # explicit path
    chain = load_option_chain("SPY")                # None if not found
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    return df


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
