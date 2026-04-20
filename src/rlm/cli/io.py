"""CLI file I/O helpers — path resolution and DataFrame loading for all commands.

Wraps ``rlm.data.readers`` and ``rlm.data.paths`` behind a CLI-friendly
interface that accepts the raw argparse values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.data.paths import get_raw_data_dir, get_processed_data_dir
from rlm.data.readers import load_bars, load_option_chain


def resolve_bars_path(symbol: str, bars_arg: str | None, data_root: str | None) -> Path:
    """Return the resolved path to the bars file.

    Parameters
    ----------
    symbol:
        Normalized ticker symbol.
    bars_arg:
        Value of the ``--bars`` CLI argument (explicit path or ``None``).
    data_root:
        Value of the ``--data-root`` CLI argument (or ``None`` to use env/default).
    """
    if bars_arg is not None:
        return Path(bars_arg).expanduser().resolve()
    return get_raw_data_dir(data_root) / f"bars_{symbol}.csv"


def resolve_chain_path(
    symbol: str, chain_arg: str | None, data_root: str | None
) -> Path | None:
    """Return the resolved path to the option chain file, or ``None`` if not specified."""
    if chain_arg is not None:
        return Path(chain_arg).expanduser().resolve()
    candidate = get_raw_data_dir(data_root) / f"option_chain_{symbol}.csv"
    return candidate if candidate.is_file() else None


def resolve_output_path(
    kind: str, symbol: str, out_arg: str | None, data_root: str | None
) -> Path:
    """Return the resolved output file path, creating its parent directory.

    Parameters
    ----------
    kind:
        Output artifact kind — e.g. ``"forecast_features"``, ``"backtest_trades"``,
        ``"backtest_equity"``.
    symbol:
        Normalized ticker symbol.
    out_arg:
        Explicit ``--out`` / ``--out-dir`` CLI value, or ``None``.
    data_root:
        Value of ``--data-root``, or ``None``.
    """
    if out_arg is not None:
        path = Path(out_arg).expanduser().resolve()
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        # Treat as directory
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{kind}_{symbol}.csv"

    processed = get_processed_data_dir(data_root)
    return processed / f"{kind}_{symbol}.csv"


def load_bars_dataframe(path: Path) -> pd.DataFrame:
    """Load a bars CSV from *path* and return a timestamp-indexed DataFrame."""
    return load_bars(symbol="", bars_path=path)


def load_option_chain_dataframe(path: Path) -> pd.DataFrame | None:
    """Load a chain CSV from *path*, or return ``None`` if the file is absent."""
    if not path.is_file():
        return None
    return load_option_chain(symbol="", chain_path=path)
