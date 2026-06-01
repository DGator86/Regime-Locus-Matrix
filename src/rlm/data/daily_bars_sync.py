"""Keep ``data/raw/bars_{SYMBOL}.csv`` aligned with live/yfinance and the 1m EODHD lake."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from rlm.data.liquidity_universe import EXPANDED_LIQUID_UNIVERSE
from rlm.data.paths import get_data_root, get_raw_data_dir, get_rlm_runtime_root
from rlm.data.rolling_store import RollingBarsStore, RollingUpdateResult
from rlm.data.stock_bars_provider import load_stock_1m_from_lake

_log = logging.getLogger(__name__)


def _repo_root(data_root: str | Path | None) -> Path:
    """Project root that contains ``data/stocks`` (not the ``data/`` folder itself)."""
    if data_root is None:
        return get_rlm_runtime_root()
    data = Path(data_root).expanduser().resolve()
    if (data / "raw").is_dir() or (data / "stocks").is_dir():
        return data.parent
    return data


@dataclass(frozen=True)
class DailyBarsSyncResult:
    symbol: str
    yfinance: RollingUpdateResult | None
    merged_from_1m: bool
    total_rows: int
    last_timestamp: pd.Timestamp | None
    bars_path: Path


def daily_from_1m_lake(symbol: str, *, data_root: str | Path | None = None) -> pd.DataFrame:
    """Resample ``data/stocks/{SYM}/1m/*_full_1m.parquet`` to daily OHLCV."""
    repo = _repo_root(data_root)
    lake = load_stock_1m_from_lake(symbol, root=repo, lookback_days=None)
    if lake.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

    frame = lake.set_index("timestamp")
    daily = (
        frame.resample("1D")
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
        .dropna(subset=["close"])
        .reset_index()
    )
    daily["timestamp"] = pd.to_datetime(daily["timestamp"]).dt.normalize()
    return daily


def _last_bar_date(df: pd.DataFrame) -> date | None:
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    return pd.Timestamp(df["timestamp"].max()).date()


def refresh_symbol_daily_bars(
    symbol: str,
    data_root: str | Path | None = None,
    *,
    today: date | None = None,
) -> DailyBarsSyncResult:
    """Refresh one symbol: yfinance append, then merge newer 1m-lake dailies if present."""
    sym = symbol.upper().strip()
    data = get_data_root(data_root)
    store = RollingBarsStore(sym, data_root=data)
    yf_result: RollingUpdateResult | None = None
    try:
        yf_result = store.update(today=today)
    except Exception as exc:
        _log.warning("RollingBarsStore[%s] yfinance update failed: %s", sym, exc)

    existing = store.load()
    lake_daily = daily_from_1m_lake(sym, data_root=data)
    merged_from_1m = False

    if not lake_daily.empty:
        lake_last = _last_bar_date(lake_daily)
        csv_last = _last_bar_date(existing) if existing is not None else None
        if lake_last is not None and (csv_last is None or lake_last > csv_last):
            parts = [existing, lake_daily] if existing is not None and not existing.empty else [lake_daily]
            combined = pd.concat(parts, ignore_index=True)
            combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.normalize()
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
            store.save(combined)
            merged_from_1m = True
            existing = combined

    total = len(existing) if existing is not None else 0
    last_ts = pd.Timestamp(existing["timestamp"].max()) if existing is not None and not existing.empty else None
    return DailyBarsSyncResult(
        symbol=sym,
        yfinance=yf_result,
        merged_from_1m=merged_from_1m,
        total_rows=total,
        last_timestamp=last_ts,
        bars_path=store.bars_path,
    )


def refresh_universe_daily_bars(
    symbols: tuple[str, ...] | list[str] | None = None,
    data_root: str | Path | None = None,
    *,
    today: date | None = None,
) -> list[DailyBarsSyncResult]:
    """Refresh daily CSV bars for the liquid universe (default: EXPANDED_LIQUID_UNIVERSE)."""
    tickers = list(symbols) if symbols is not None else list(EXPANDED_LIQUID_UNIVERSE)
    out: list[DailyBarsSyncResult] = []
    for sym in tickers:
        try:
            out.append(refresh_symbol_daily_bars(sym, data_root, today=today))
        except Exception as exc:
            _log.error("daily bars sync failed for %s: %s", sym, exc)
    return out


def load_best_daily_bars(
    symbol: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return the freshest daily bars available without network I/O."""
    sym = symbol.upper().strip()
    data = get_data_root(data_root)
    csv_path = get_raw_data_dir(data) / f"bars_{sym}.csv"

    candidates: list[tuple[date, pd.DataFrame]] = []

    if csv_path.is_file():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp")
        if not df.empty:
            last = _last_bar_date(df)
            if last is not None:
                candidates.append((last, df))

    lake_daily = daily_from_1m_lake(sym, data_root=data)
    if not lake_daily.empty:
        last = _last_bar_date(lake_daily)
        if last is not None:
            candidates.append((last, lake_daily))

    if not candidates:
        raise FileNotFoundError(f"No daily bars for {sym} (checked {csv_path} and 1m lake)")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1].reset_index(drop=True)
