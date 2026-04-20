"""Data backend selector for RLM.

Controls how the runtime resolves data sources.  The default (``AUTO``) prefers
Parquet lake artifacts when they exist and falls back to CSV transparently.

Usage::

    from rlm.data.backend import DataBackend, resolve_backend

    backend = resolve_backend("auto", symbol="SPY", data_root="/mnt/lake")
    # → DataBackend.LAKE  (if parquet found)
    # → DataBackend.CSV   (if only csv found)
    # → raises FileNotFoundError if neither exists
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from rlm.data.paths import get_data_root, get_raw_data_dir


class DataBackend(str, Enum):
    AUTO = "auto"
    CSV = "csv"
    LAKE = "lake"


def resolve_backend(
    backend: str | DataBackend,
    symbol: str,
    data_root: str | Path | None = None,
    interval: str | None = None,
) -> DataBackend:
    """Resolve the effective backend for *symbol*.

    AUTO logic:
      1. Parquet lake artifacts present → LAKE
      2. CSV file present → CSV
      3. Neither → DataBackend.CSV (let caller raise a clean FileNotFoundError)
    """
    b = DataBackend(backend)

    if b != DataBackend.AUTO:
        return b

    # Check lake
    root = get_data_root(data_root)
    lake_candidates = [
        root / "lake" / symbol.upper(),
        root / "stocks" / symbol.upper(),
        root / f"bars_{symbol.upper()}.parquet",
    ]
    if interval:
        lake_candidates.insert(0, root / "lake" / symbol.upper() / interval)

    if any(p.exists() for p in lake_candidates):
        return DataBackend.LAKE

    # Check CSV
    csv_path = get_raw_data_dir(data_root) / f"bars_{symbol.upper()}.csv"
    if csv_path.is_file():
        return DataBackend.CSV

    # Default to CSV so downstream raises a consistent FileNotFoundError
    return DataBackend.CSV
