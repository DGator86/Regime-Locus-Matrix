"""Backward-compatible shim for IBKR stock parquet writes."""

from rlm.ingestion.writers import write_ibkr_stock_parquet as fetch_stock_to_parquet

__all__ = ["fetch_stock_to_parquet"]
