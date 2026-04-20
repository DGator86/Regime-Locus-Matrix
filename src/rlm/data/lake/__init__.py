"""Parquet/DuckDB data lake readers and writers for RLM.

Lake layout (under the data root)::

    data/
    ├── lake/
    │   ├── {SYMBOL}/
    │   │   ├── bars_{interval}.parquet
    │   │   └── option_chain_{date}.parquet
    │   └── metadata.json
    ├── raw/          ← CSV fallback / ingest staging
    └── processed/   ← forecast/backtest outputs
"""

from rlm.data.lake.readers import load_bars_parquet, load_option_chain_parquet
from rlm.data.lake.writers import write_bars_parquet, write_option_chain_parquet
from rlm.data.lake.metadata import LakeMetadata

__all__ = [
    "load_bars_parquet",
    "load_option_chain_parquet",
    "write_bars_parquet",
    "write_option_chain_parquet",
    "LakeMetadata",
]
