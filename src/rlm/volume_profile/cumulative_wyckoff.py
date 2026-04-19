"""Cumulative Wyckoff-style effort/result divergence signals."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    return work.dropna(subset=["high", "low", "close", "volume"])


def cumulative_effort_result(df: pd.DataFrame, start_idx: int = 0) -> float:
    """Return a divergence score in ``[-1, 1]`` for cumulative effort vs result."""
    work = _prep(df)
    if work.empty or len(work) <= start_idx:
        return 0.0

    seg = work.iloc[int(start_idx) :].copy()
    seg["cum_volume"] = seg["volume"].cumsum()
    seg["cum_range"] = (seg["high"] - seg["low"]).cumsum()

    bar_range = (seg["high"] - seg["low"]).clip(lower=1e-12)
    vol_pct = float(seg["volume"].rank(pct=True).iloc[-1])
    range_pct = float(bar_range.rank(pct=True).iloc[-1])
    imbalance = vol_pct - range_pct

    move = float(seg["close"].iloc[-1] - seg["close"].iloc[0])
    if abs(imbalance) < 0.10:
        return 0.0
    if move <= 0 and imbalance > 0:
        return min(1.0, abs(imbalance))
    if move >= 0 and imbalance > 0:
        return -min(1.0, abs(imbalance))
    return 0.0


def session_cumulative_divergence(df: pd.DataFrame, session_start_time: datetime) -> pd.Series:
    """Return a per-bar cumulative divergence series for a session."""
    work = _prep(df)
    if work.empty:
        return pd.Series(dtype=float)

    if "timestamp" in work.columns:
        start_ts = pd.Timestamp(session_start_time)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        work = work.loc[work["timestamp"] >= start_ts]

    scores = [cumulative_effort_result(work.iloc[: i + 1], start_idx=0) for i in range(len(work))]
    return pd.Series(scores, index=work.index, name="cumulative_wyckoff_score")


def detect_absorption_climax(df: pd.DataFrame, threshold: float = 0.8) -> list[datetime]:
    """Identify timestamps where absolute cumulative divergence exceeds threshold."""
    work = _prep(df)
    if work.empty:
        return []

    if "timestamp" in work.columns:
        session_start = pd.Timestamp(work["timestamp"].min())
    else:
        session_start = pd.Timestamp.utcnow()
    series = session_cumulative_divergence(work, session_start)
    peaks = work.loc[series.abs() >= float(threshold)]

    if "timestamp" not in peaks.columns:
        return []
    return [ts.to_pydatetime() for ts in pd.to_datetime(peaks["timestamp"], utc=True)]
