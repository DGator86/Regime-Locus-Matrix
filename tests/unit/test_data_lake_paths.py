from datetime import date
from pathlib import Path

from rlm.datasets.data_lake import (
    data_lake_root,
    option_ticker_file_slug,
    options_flatfile_daily_parquet,
    stock_1d_parquet,
    options_contracts_dir,
)


def test_data_lake_paths() -> None:
    root = Path("/repo")
    assert data_lake_root(root) == Path("/repo/data")
    p = stock_1d_parquet("SPY", duration_slug="2y", root=root)
    assert p == Path("/repo/data/stocks/SPY/1d/spy_2y_1d.parquet")
    assert "contracts" in str(options_contracts_dir("QQQ", root=root))


def test_option_slug() -> None:
    assert "SPY260619C00650000" in option_ticker_file_slug("O:SPY260619C00650000")


def test_options_flatfile_parquet_path() -> None:
    root = Path("/repo")
    p = options_flatfile_daily_parquet("SPY", "trades", date(2025, 1, 7), root=root)
    assert p == Path("/repo/data/options/SPY/flatfiles/trades/2025-01-07.parquet")
