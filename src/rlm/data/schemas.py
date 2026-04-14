"""PyArrow schemas + lightweight validation helpers for lake tables."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, table_name: str = "table") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")
