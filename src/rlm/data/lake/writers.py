from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)


def save_parquet_versioned(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    source: str = "unknown",
    schema: str = "unknown",
    lineage_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Save parquet and append immutable lineage metadata for reproducibility."""
    save_parquet(df, path, index=index)
    file_bytes = Path(path).read_bytes()
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "source": source,
        "schema": schema,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sha256": sha256,
    }
    out_root = lineage_root or Path(path).parents[3]
    lineage_path = out_root / "metadata" / "lineage" / "lineage_log.jsonl"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with lineage_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record
