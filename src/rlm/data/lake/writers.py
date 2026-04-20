"""Parquet lake writers for RLM bars and option chains."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def write_bars_parquet(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "1d",
    data_root: str | Path | None = None,
) -> Path:
    """Write OHLCV bars to the Parquet lake.

    Writes to ``<data_root>/lake/<SYMBOL>/bars_<interval>.parquet``.

    Parameters
    ----------
    df:
        OHLCV DataFrame (timestamp-indexed or with a ``timestamp`` column).
    symbol:
        Ticker symbol.
    interval:
        Bar interval string.
    data_root:
        Override data root directory.

    Returns
    -------
    Path
        The written Parquet file path.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyarrow is required to write Parquet files.\n"
            "Install: pip install -e '.[datalake]'"
        )

    sym = symbol.upper()
    root = get_data_root(data_root)
    lake_dir = root / "lake" / sym
    lake_dir.mkdir(parents=True, exist_ok=True)

    out_path = lake_dir / f"bars_{interval}.parquet"

    reset = df.copy()
    if reset.index.name == "timestamp":
        reset = reset.reset_index()

    reset.to_parquet(out_path, index=False)
    log.info("lake write bars  symbol=%s interval=%s rows=%d path=%s", sym, interval, len(reset), out_path)
    return out_path


def write_option_chain_parquet(
    df: pd.DataFrame,
    symbol: str,
    as_of: str,
    data_root: str | Path | None = None,
) -> Path:
    """Write an option chain snapshot to the Parquet lake.

    Writes to ``<data_root>/options/<SYMBOL>/option_chain_<as_of>.parquet``.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyarrow is required to write Parquet files.\n"
            "Install: pip install -e '.[datalake]'"
        )

    sym = symbol.upper()
    root = get_data_root(data_root)
    chain_dir = root / "options" / sym
    chain_dir.mkdir(parents=True, exist_ok=True)

    out_path = chain_dir / f"option_chain_{as_of}.parquet"
    df.to_parquet(out_path, index=False)
    log.info("lake write chain  symbol=%s as_of=%s rows=%d path=%s", sym, as_of, len(df), out_path)
    return out_path
