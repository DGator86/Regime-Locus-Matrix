"""Pull IBKR equity bars and write Parquet under ``data/stocks/{SYM}/``."""

from __future__ import annotations

from pathlib import Path

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.datasets.data_lake import stock_1d_parquet, stock_1m_parquet
from rlm.datasets.data_lake_io import save_parquet


def fetch_stock_to_parquet(
    symbol: str,
    *,
    duration: str,
    bar_size: str,
    duration_slug: str,
    out_path: Path | None = None,
    interval_dir: str = "1d",
    repo_root: Path | None = None,
    end_datetime: str = "",
    timeout_sec: float = 180.0,
) -> Path:
    """
    Fetch via ``ibapi`` and save. ``interval_dir`` is ``\"1d\"`` or ``\"1m\"`` for path layout.
    ``duration_slug`` is a short filesystem token (e.g. ``2y``, ``10d``).
    """
    df = fetch_historical_stock_bars(
        symbol,
        duration=str(duration),
        bar_size=str(bar_size),
        end_datetime=end_datetime,
        timeout_sec=timeout_sec,
    )
    if out_path is None:
        if interval_dir == "1m":
            out_path = stock_1m_parquet(symbol, duration_slug=duration_slug, root=repo_root)
        else:
            out_path = stock_1d_parquet(symbol, duration_slug=duration_slug, root=repo_root)
    save_parquet(df, Path(out_path))
    return Path(out_path)
