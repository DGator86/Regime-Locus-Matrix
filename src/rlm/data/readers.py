"""Centralised data reader abstraction for RLM.

CLI commands and services must load bars and option chains through these
functions rather than calling ``pd.read_csv`` directly.  This keeps the
loading contract in one place and allows future evolution to Parquet/DuckDB
without touching CLI or service code.

Backend selection (``backend`` parameter or ``--backend`` CLI flag):

    auto  Prefer lake (Parquet) when present, fall back to csv silently.
    csv   Always read CSV from the raw/ directory.
    lake  Always read Parquet from the lake/ directory.

Usage::

    from rlm.data.readers import load_bars, load_option_chain

    bars = load_bars("SPY")                        # auto backend
    bars = load_bars("SPY", backend="csv")         # force CSV
    bars = load_bars("SPY", backend="lake")        # force Parquet
    chain = load_option_chain("SPY")               # None if not found
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.backend import DataBackend, resolve_backend
from rlm.data.paths import get_raw_data_dir
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def load_bars(
    symbol: str,
    bars_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str | DataBackend = DataBackend.AUTO,
    interval: str = "1d",
) -> pd.DataFrame:
    """Load OHLCV bars for *symbol*, returning a timestamp-indexed DataFrame.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    bars_path:
        Explicit file path.  When provided, *backend* and *data_root* are ignored.
    data_root:
        Override data root (see ``rlm.data.paths.get_data_root``).
    backend:
        ``"auto"`` (default), ``"csv"``, or ``"lake"``.
    interval:
        Bar interval used for lake path resolution (e.g. ``"1d"``, ``"1h"``).

    Raises
    ------
    FileNotFoundError
        When the resolved file does not exist.
    """
    # Explicit path bypasses backend selection entirely
    if bars_path is not None:
        return _load_csv(Path(bars_path).expanduser().resolve(), symbol)

    effective = resolve_backend(backend, symbol, data_root, interval)

    if effective == DataBackend.LAKE:
        try:
            from rlm.data.lake.readers import load_bars_parquet
            return load_bars_parquet(symbol, interval=interval, data_root=data_root)
        except (FileNotFoundError, ImportError) as exc:
            log.debug("lake bars failed (%s), falling back to CSV", exc)
            effective = DataBackend.CSV

    # CSV path
    path = get_raw_data_dir(data_root) / f"bars_{symbol.upper()}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Bars file not found: {path}\n"
            f"Fetch data first:  rlm ingest --symbol {symbol}\n"
            f"Or set a different root:  --data-root /your/path  or  RLM_DATA_ROOT=/your/path"
        )
    return _load_csv(path, symbol)


def load_option_chain(
    symbol: str,
    chain_path: str | Path | None = None,
    data_root: str | Path | None = None,
    backend: str | DataBackend = DataBackend.AUTO,
    as_of: str | None = None,
) -> pd.DataFrame | None:
    """Load an option chain snapshot for *symbol*, or return ``None`` if absent.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    chain_path:
        Explicit file path.  When provided, *backend* is ignored.
    data_root:
        Override data root directory.
    backend:
        ``"auto"``, ``"csv"``, or ``"lake"``.
    as_of:
        Date string ``YYYY-MM-DD`` for lake backend snapshot selection.
    """
    if chain_path is not None:
        p = Path(chain_path).expanduser().resolve()
        if not p.is_file():
            return None
        return _load_chain_csv(p)

    effective = resolve_backend(backend, symbol, data_root)

    if effective == DataBackend.LAKE:
        try:
            from rlm.data.lake.readers import load_option_chain_parquet
            return load_option_chain_parquet(symbol, as_of=as_of, data_root=data_root)
        except (FileNotFoundError, ImportError):
            effective = DataBackend.CSV

    candidate = get_raw_data_dir(data_root) / f"option_chain_{symbol.upper()}.csv"
    if not candidate.is_file():
        log.debug("load_option_chain: no CSV for %s", symbol)
        return None
    return _load_chain_csv(candidate)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _load_csv(path: Path, symbol: str) -> pd.DataFrame:
    log.info("load_bars csv  symbol=%s path=%s", symbol.upper(), path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _load_chain_csv(path: Path) -> pd.DataFrame:
    log.info("load_option_chain csv  path=%s", path)
    date_cols = [c for c in ["timestamp", "expiry"] if True]
    try:
        return pd.read_csv(path, parse_dates=date_cols)
    except Exception as exc:
        log.warning("load_option_chain failed to parse %s: %s", path, exc)
        return None  # type: ignore[return-value]
