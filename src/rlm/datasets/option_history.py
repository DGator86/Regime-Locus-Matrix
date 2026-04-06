"""Append / merge Massive option chain snapshots into a historical CSV for backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _dedupe_subset_columns(df: pd.DataFrame) -> list[str]:
    base = ["timestamp", "underlying", "expiry", "strike", "option_type"]
    if "contract_symbol" in df.columns and df["contract_symbol"].notna().any():
        return ["timestamp", "contract_symbol"]
    return base


def merge_option_chain_history(
    existing: pd.DataFrame | None,
    new_rows: pd.DataFrame,
    *,
    replace_calendar_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Concatenate ``new_rows`` onto ``existing``, optionally dropping prior rows on the same calendar date.

    Deduplicates on (timestamp, contract_symbol) when ``contract_symbol`` is present and non-null,
    else on (timestamp, underlying, expiry, strike, option_type).
    """
    if new_rows.empty:
        return existing.copy() if existing is not None and not existing.empty else pd.DataFrame()

    new_rows = new_rows.copy()
    new_rows["timestamp"] = pd.to_datetime(new_rows["timestamp"])

    if existing is None or existing.empty:
        out = new_rows
    else:
        old = existing.copy()
        old["timestamp"] = pd.to_datetime(old["timestamp"])
        if replace_calendar_date is not None:
            d = pd.Timestamp(replace_calendar_date).normalize().date()
            mask = old["timestamp"].dt.date != d
            old = old.loc[mask]
        out = pd.concat([old, new_rows], ignore_index=True)

    subset = _dedupe_subset_columns(out)
    out = out.drop_duplicates(subset=subset, keep="last")
    sort_cols = [c for c in ("timestamp", "expiry", "strike", "option_type") if c in out.columns]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def read_option_chain_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp", "expiry"])


def write_option_chain_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
