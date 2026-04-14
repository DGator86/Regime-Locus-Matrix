"""Lake-level utilities (append/query/compact helpers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_frame(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    return pd.concat([existing, incoming], ignore_index=True)


def query_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def compact_parquet(paths: list[Path], out_path: Path) -> Path:
    frames = [pd.read_parquet(p) for p in paths if Path(p).exists()]
    if not frames:
        raise ValueError("No parquet files supplied for compaction")
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path
