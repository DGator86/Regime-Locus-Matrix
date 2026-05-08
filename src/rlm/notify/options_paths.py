"""Resolve paths for the options dry-run monitor CSV (marks / EOD / health / dashboard).

The master stack may write ``--options-trade-log`` to a non-default filename (e.g.
``options_large_account_trade_log.csv``). Readers that only checked
``data/processed/trade_log.csv`` would see an empty file while the real log grew
elsewhere. ``RLM_OPTIONS_TRADE_LOG_PATH`` wins when set (same as monitor flag).
"""

from __future__ import annotations

import os
from pathlib import Path


def options_trade_log_primary(root: Path) -> Path:
    """Canonical primary path (env override or default ``trade_log.csv``)."""
    raw = (os.environ.get("RLM_OPTIONS_TRADE_LOG_PATH") or "").strip()
    if not raw:
        return root / "data" / "processed" / "trade_log.csv"
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def options_trade_log_read_paths(root: Path) -> list[Path]:
    """Paths to try when *reading* monitor rows (primary first, then common fallback)."""
    primary = options_trade_log_primary(root)
    fb = root / "data" / "processed" / "options_large_account_trade_log.csv"
    out: list[Path] = [primary]
    try:
        if fb.resolve() != primary.resolve():
            out.append(fb)
    except OSError:
        out.append(fb)
    return out


def options_trade_log_mtime_paths(root: Path) -> list[Path]:
    """Same as :func:`options_trade_log_read_paths` (alias for health / staleness)."""
    return options_trade_log_read_paths(root)
