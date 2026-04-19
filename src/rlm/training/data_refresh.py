from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_symbol_feature_frames(symbols: list[str], data_dir: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = Path(data_dir)
    for symbol in symbols:
        path = base / f"features_{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file for {symbol}: {path}")
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def count_new_rows_since(df: pd.DataFrame, trained_at: str) -> int:
    if "timestamp" not in df.columns:
        return 0
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    t0 = pd.Timestamp(trained_at)
    return int((ts > t0).sum())
