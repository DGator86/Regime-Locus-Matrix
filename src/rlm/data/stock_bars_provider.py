"""Load equity/index OHLCV for factors: EODHD lake (primary) with IBKR fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from rlm.data.bar_timeframes import duration_to_calendar_days, is_intraday_bar_size
from rlm.data.eodhd_stocks import fetch_intraday_lookback_days, load_eodhd_api_key
from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.lake import stock_1m_parquet
from rlm.datasets.backtest_data import fetch_ibkr_intraday_bars_range


def stock_bars_source() -> str:
    """``eodhd`` (lake + optional live backfill), ``ibkr``, or ``auto`` (lake then IBKR)."""
    return (os.environ.get("RLM_STOCK_BARS_SOURCE") or "eodhd").strip().lower()


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _lake_path(symbol: str, *, root: Path) -> Path:
    return stock_1m_parquet(symbol, duration_slug="full", root=root)


def load_stock_1m_from_lake(
    symbol: str,
    *,
    root: Path,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    path = _lake_path(symbol, root=root)
    if not path.is_file():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if lookback_days is not None and lookback_days > 0:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(lookback_days))
        df = df.loc[df["timestamp"] >= cutoff]
    return df.reset_index(drop=True)


def export_chart_bars_csv(
    symbol: str,
    *,
    root: Path,
    tail_bars: int = 120,
) -> Path | None:
    """Write last N 1m bars for dashboard charts (``data/processed/chart_bars_{SYM}.csv``)."""
    sym = str(symbol).strip().upper()
    lake = load_stock_1m_from_lake(sym, root=root, lookback_days=30)
    if lake.empty:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    out_path = root / "data" / "processed" / f"chart_bars_{sym.lower()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lake[cols].tail(int(tail_bars)).to_csv(out_path, index=False)
    return out_path


def merge_bars_into_lake(
    symbol: str,
    fresh: pd.DataFrame,
    *,
    root: Path,
) -> pd.DataFrame:
    """Merge ``fresh`` bars into ``data/stocks/{SYM}/1m/*_full_1m.parquet``."""
    cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    path = _lake_path(symbol, root=root)
    existing = load_stock_1m_from_lake(symbol, root=root, lookback_days=None)
    ex = existing[cols].copy() if not existing.empty else pd.DataFrame(columns=cols)
    fr = fresh[cols].copy() if not fresh.empty else pd.DataFrame(columns=cols)
    merged = pd.concat([ex, fr], ignore_index=True)
    if merged.empty:
        return merged
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def fetch_stock_bars(
    symbol: str,
    *,
    duration: str,
    bar_size: str,
    data_root: Path,
    serialize_ibkr: bool = False,
    ibkr_lock: object | None = None,
) -> pd.DataFrame:
    """Primary stock bar loader for live universe / regime pipelines."""
    sym = str(symbol).strip().upper()
    source = stock_bars_source()
    days = duration_to_calendar_days(duration)

    if is_intraday_bar_size(bar_size):
        if source in ("eodhd", "auto"):
            lake = load_stock_1m_from_lake(sym, root=data_root, lookback_days=days)
            min_bars = int(os.environ.get("RLM_EODHD_MIN_LAKE_BARS", "500"))
            if len(lake) >= min_bars:
                return lake
            if source == "eodhd" and load_eodhd_api_key():
                try:
                    fresh = fetch_intraday_lookback_days(sym, lookback_days=days)
                    if not fresh.empty:
                        return merge_bars_into_lake(sym, fresh, root=data_root)
                except Exception:
                    if not _env_truthy("RLM_ALLOW_IBKR_STOCK_BARS"):
                        return lake
            if source == "auto" and not lake.empty:
                return lake

        return _fetch_ibkr_intraday(sym, duration=duration, bar_size=bar_size, serialize=serialize_ibkr, lock=ibkr_lock)

    if source == "eodhd":
        # Daily regime on EODHD-only hosts: prefer multi-year daily CSV / full-lake
        # resample. Truncating the 1m lake to ``duration`` (often 30 D) yields ~21
        # sessions — too short for live vol_window baselines.
        if not _env_truthy("RLM_ALLOW_IBKR_STOCK_BARS"):
            daily = _load_daily_bars_offline(sym, data_root=data_root, lookback_days=days)
            if not daily.empty:
                return daily

    return _fetch_ibkr_daily(sym, duration=duration, bar_size=bar_size, serialize=serialize_ibkr, lock=ibkr_lock)


def _data_dir_for_daily(data_root: Path) -> Path:
    """Normalize repo root vs ``data/`` for :func:`load_best_daily_bars`."""
    root = Path(data_root).expanduser().resolve()
    if (root / "data" / "raw").is_dir() or (root / "data" / "stocks").is_dir():
        return root / "data"
    return root


def _trim_daily_lookback(daily: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
    if daily.empty or lookback_days <= 0:
        return daily.reset_index(drop=True)
    out = daily.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    span = pd.Timedelta(days=int(lookback_days))
    trimmed_now = out.loc[out["timestamp"] >= (pd.Timestamp.now().normalize() - span)]
    last = pd.Timestamp(out["timestamp"].max())
    trimmed_last = out.loc[out["timestamp"] >= (last.normalize() - span)]
    # Stale CSVs often leave a sparse "now" window; prefer the denser trailing span.
    if len(trimmed_last) > len(trimmed_now):
        trimmed = trimmed_last
    else:
        trimmed = trimmed_now if not trimmed_now.empty else trimmed_last
    return (trimmed if not trimmed.empty else out).reset_index(drop=True)


def _load_daily_bars_offline(
    symbol: str,
    *,
    data_root: Path,
    lookback_days: int,
) -> pd.DataFrame:
    """Load daily OHLCV without IBKR (best CSV/lake), then apply lookback."""
    # Lazy import: daily_bars_sync imports load_stock_1m_from_lake from this module.
    from rlm.data.daily_bars_sync import load_best_daily_bars

    try:
        daily = load_best_daily_bars(symbol, data_root=_data_dir_for_daily(data_root))
    except FileNotFoundError:
        daily = pd.DataFrame()
    if daily.empty:
        lake = load_stock_1m_from_lake(symbol, root=data_root, lookback_days=None)
        if lake.empty:
            return daily
        daily = (
            lake.set_index("timestamp")
            .resample("1D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "vwap": "mean",
                }
            )
            .dropna(how="all")
            .reset_index()
        )
    return _trim_daily_lookback(daily, lookback_days=lookback_days)


def _fetch_ibkr_intraday(
    symbol: str,
    *,
    duration: str,
    bar_size: str,
    serialize: bool,
    lock: object | None,
) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    days = duration_to_calendar_days(duration)
    start = end - pd.Timedelta(days=days)

    def _pull() -> pd.DataFrame:
        return fetch_ibkr_intraday_bars_range(symbol, start, end, bar_size=bar_size, timeout_sec=180.0)

    if serialize and lock is not None:
        with lock:  # type: ignore[union-attr]
            return _pull()
    return _pull()


def _fetch_ibkr_daily(
    symbol: str,
    *,
    duration: str,
    bar_size: str,
    serialize: bool,
    lock: object | None,
) -> pd.DataFrame:
    fetch_kw: dict[str, object] = {
        "duration": duration,
        "bar_size": bar_size,
        "timeout_sec": 120.0,
    }

    def _pull() -> pd.DataFrame:
        return fetch_historical_stock_bars(symbol, **fetch_kw)

    if serialize and lock is not None:
        with lock:  # type: ignore[union-attr]
            return _pull()
    return _pull()
