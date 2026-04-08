"""Stream gzip CSV flat files → filtered Parquet (options research store)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from rlm.datasets.data_lake_io import save_parquet

_NS_TS_COLUMNS = ("sip_timestamp", "participant_timestamp", "window_start", "timestamp")


def _attach_utc_from_ns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _NS_TS_COLUMNS:
        if col in out.columns and out[col].notna().any():
            try:
                out[f"{col}_utc"] = pd.to_datetime(out[col], unit="ns", utc=True)
            except (ValueError, TypeError):
                pass
    return out


def filter_ticker_prefixes(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    if "ticker" not in df.columns:
        return df
    pre = [str(p).strip().upper() for p in prefixes if str(p).strip()]
    if not pre:
        return df
    s = df["ticker"].astype(str).str.upper()
    mask = s.str.startswith(pre[0].upper())
    for p in pre[1:]:
        mask |= s.str.startswith(p.upper())
    return df.loc[mask]


def gzip_csv_to_filtered_parquet(
    gz_path: Path,
    out_parquet: Path,
    *,
    ticker_prefixes: list[str] | None,
    chunksize: int = 400_000,
) -> int:
    """
    Read ``.csv.gz`` in chunks, optional ticker prefix filter, write one Parquet file.
    Returns row count written.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("Parquet streaming requires pyarrow.") from e

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(gz_path, chunksize=chunksize, compression="gzip", low_memory=False)
    writer: pq.ParquetWriter | None = None
    total = 0

    for chunk in reader:
        if ticker_prefixes:
            chunk = filter_ticker_prefixes(chunk, ticker_prefixes)
        if chunk.empty:
            continue
        chunk = _attach_utc_from_ns(chunk)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_parquet), table.schema)
        writer.write_table(table)
        total += len(chunk)

    if writer is not None:
        writer.close()

    if total == 0:
        # still produce a small marker file for pipeline idempotency
        empty = pd.DataFrame()
        save_parquet(empty, out_parquet)

    return total