"""Load equity/index OHLCV for factors: EODHD lake (primary) with IBKR fallback."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from rlm.data.bar_timeframes import duration_to_calendar_days, is_intraday_bar_size
from rlm.data.eodhd_stocks import fetch_intraday_lookback_days, load_eodhd_api_key
from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.lake import save_parquet, stock_1m_parquet
from rlm.datasets.backtest_data import fetch_ibkr_intraday_bars_range

try:  # pragma: no cover - exercised on POSIX hosts in integration.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]


_STOCK_1M_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


def stock_bars_source() -> str:
    """``eodhd`` (lake + optional live backfill), ``ibkr``, or ``auto`` (lake then IBKR)."""
    return (os.environ.get("RLM_STOCK_BARS_SOURCE") or "eodhd").strip().lower()


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _lake_path(symbol: str, *, root: Path) -> Path:
    return stock_1m_parquet(symbol, duration_slug="full", root=root)


def _empty_1m_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=_STOCK_1M_COLUMNS)


@contextmanager
def _lake_update_lock(path: Path):
    """Serialize read-merge-write updates for one lake parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    if fcntl is not None:
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return

    lock_fd: int | None = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def load_stock_1m_from_lake(
    symbol: str,
    *,
    root: Path,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    path = _lake_path(symbol, root=root)
    if not path.is_file():
        return _empty_1m_bars()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return _empty_1m_bars()
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
    path = _lake_path(symbol, root=root)
    with _lake_update_lock(path):
        existing = load_stock_1m_from_lake(symbol, root=root, lookback_days=None)
        ex = existing[_STOCK_1M_COLUMNS].copy() if not existing.empty else _empty_1m_bars()
        fr = fresh[_STOCK_1M_COLUMNS].copy() if not fresh.empty else _empty_1m_bars()
        merged = pd.concat([ex, fr], ignore_index=True)
        if merged.empty:
            return merged
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
        merged = merged.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        save_parquet(merged, path, index=False)
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
            allow_ibkr = _env_truthy("RLM_ALLOW_IBKR_STOCK_BARS")
            if load_eodhd_api_key():
                try:
                    fresh = fetch_intraday_lookback_days(sym, lookback_days=days)
                    if not fresh.empty:
                        lake = merge_bars_into_lake(sym, fresh, root=data_root)
                        if len(lake) >= min_bars:
                            return lake
                except Exception:
                    if source == "eodhd" and not allow_ibkr:
                        return _empty_1m_bars()
            if source == "eodhd" and not allow_ibkr:
                return _empty_1m_bars()

        return _fetch_ibkr_intraday(sym, duration=duration, bar_size=bar_size, serialize=serialize_ibkr, lock=ibkr_lock)

    if source == "eodhd":
        # Daily regime on EODHD-only hosts: fall through to IBKR unless allowed
        if not _env_truthy("RLM_ALLOW_IBKR_STOCK_BARS"):
            lake = load_stock_1m_from_lake(sym, root=data_root, lookback_days=days)
            if not lake.empty:
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
                return daily

    return _fetch_ibkr_daily(sym, duration=duration, bar_size=bar_size, serialize=serialize_ibkr, lock=ibkr_lock)


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
