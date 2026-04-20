"""Parquet/DuckDB lake readers for RLM bars and option chains."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def load_bars_parquet(
    symbol: str,
    interval: str = "1d",
    data_root: str | Path | None = None,
    use_duckdb: bool = False,
) -> pd.DataFrame:
    """Load OHLCV bars from the Parquet lake.

    Searches for bars in this order:
      1. ``<data_root>/lake/<SYMBOL>/bars_<interval>.parquet``
      2. ``<data_root>/stocks/<SYMBOL>/<interval>/*.parquet``  (IBKR layout)
      3. ``<data_root>/bars_<SYMBOL>.parquet``  (flat fallback)

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    interval:
        Bar interval string (e.g. ``"1d"``, ``"1h"``, ``"5m"``).
    data_root:
        Override data root directory.
    use_duckdb:
        When True and ``duckdb`` is installed, query via DuckDB for
        efficient partition scanning on large datasets.
    """
    sym = symbol.upper()
    root = get_data_root(data_root)

    candidates: list[Path] = [
        root / "lake" / sym / f"bars_{interval}.parquet",
        root / "lake" / sym / f"{interval}.parquet",
        root / "stocks" / sym / interval,
        root / f"bars_{sym}.parquet",
    ]

    for candidate in candidates:
        if candidate.is_file():
            log.info("lake read bars  symbol=%s interval=%s path=%s", sym, interval, candidate)
            return _read_parquet(candidate, use_duckdb=use_duckdb)
        if candidate.is_dir():
            parts = list(candidate.glob("*.parquet"))
            if parts:
                log.info(
                    "lake read bars  symbol=%s interval=%s dir=%s files=%d",
                    sym, interval, candidate, len(parts),
                )
                return _read_parquet_dir(candidate, use_duckdb=use_duckdb)

    raise FileNotFoundError(
        f"No Parquet bars found for {sym} (interval={interval}) in {root}.\n"
        f"Run: rlm ingest --symbol {sym} --source ibkr  "
        f"or use --backend csv to fall back to CSV."
    )


def load_option_chain_parquet(
    symbol: str,
    as_of: str | None = None,
    data_root: str | Path | None = None,
    use_duckdb: bool = False,
) -> pd.DataFrame | None:
    """Load an option chain snapshot from the Parquet lake, or return ``None``.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    as_of:
        Date string ``YYYY-MM-DD``.  When omitted, returns the most recent snapshot.
    data_root:
        Override data root directory.
    use_duckdb:
        Query via DuckDB when available.
    """
    sym = symbol.upper()
    root = get_data_root(data_root)
    chain_dir = root / "options" / sym

    if not chain_dir.exists():
        alt = root / "lake" / sym / "option_chain"
        if alt.exists():
            chain_dir = alt

    if not chain_dir.exists():
        log.debug("lake: no option chain dir for %s", sym)
        return None

    if as_of:
        specific = chain_dir / f"option_chain_{as_of}.parquet"
        if specific.is_file():
            log.info("lake read chain  symbol=%s as_of=%s", sym, as_of)
            return _read_parquet(specific, use_duckdb=use_duckdb)
        log.debug("lake: no chain snapshot for %s as_of=%s", sym, as_of)
        return None

    # Return most recent snapshot
    snapshots = sorted(chain_dir.glob("*.parquet"), reverse=True)
    if not snapshots:
        return None

    log.info("lake read chain  symbol=%s latest=%s", sym, snapshots[0].name)
    return _read_parquet(snapshots[0], use_duckdb=use_duckdb)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _read_parquet(path: Path, use_duckdb: bool = False) -> pd.DataFrame:
    if use_duckdb:
        try:
            import duckdb
            con = duckdb.connect(":memory:")
            df = con.execute(f"SELECT * FROM '{path}'").df()
            con.close()
            return _normalize_timestamp_index(df)
        except ImportError:
            pass  # fall through to pandas

    df = pd.read_parquet(path)
    return _normalize_timestamp_index(df)


def _read_parquet_dir(directory: Path, use_duckdb: bool = False) -> pd.DataFrame:
    if use_duckdb:
        try:
            import duckdb
            pattern = str(directory / "*.parquet")
            con = duckdb.connect(":memory:")
            df = con.execute(f"SELECT * FROM '{pattern}'").df()
            con.close()
            return _normalize_timestamp_index(df)
        except ImportError:
            pass

    parts = sorted(directory.glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    return _normalize_timestamp_index(df)


def _normalize_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").set_index("timestamp")
    elif df.index.name != "timestamp" and pd.api.types.is_datetime64_any_dtype(df.index):
        df.index.name = "timestamp"
    return df
