"""Parquet writers for the data lake (requires pyarrow)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=index)
    except ImportError as e:
        raise ImportError(
            "Parquet output requires pyarrow. Install: pip install pyarrow "
            "or pip install 'regime-locus-matrix[datalake]'"
        ) from e
