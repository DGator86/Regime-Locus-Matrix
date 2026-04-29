"""Backward-compatible shims for Massive options parquet writers."""

from rlm.ingestion.writers import (
    write_massive_option_bars_parquet as fetch_option_aggs_to_parquet,
)
from rlm.ingestion.writers import (
    write_massive_option_contracts_parquet as fetch_option_contracts_to_parquet,
)
from rlm.ingestion.writers import (
    write_massive_option_quotes_parquet as fetch_option_quotes_to_parquet,
)
from rlm.ingestion.writers import (
    write_massive_option_trades_parquet as fetch_option_trades_to_parquet,
)

__all__ = [
    "fetch_option_contracts_to_parquet",
    "fetch_option_aggs_to_parquet",
    "fetch_option_trades_to_parquet",
    "fetch_option_quotes_to_parquet",
]
