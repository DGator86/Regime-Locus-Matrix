"""Backward-compatible shim to storage helpers in :mod:`rlm.data.lake`."""

from rlm.data.lake import save_parquet

__all__ = ["save_parquet"]
