"""Tests for option chain usability guard used by CSV/parquet readers."""

from __future__ import annotations

import pandas as pd

from rlm.data.option_chain import REQUIRED_CHAIN_COLUMNS, option_chain_is_usable


def test_option_chain_is_usable_full_schema() -> None:
    row = {c: 0 for c in REQUIRED_CHAIN_COLUMNS}
    df = pd.DataFrame([row])
    assert option_chain_is_usable(df) is True


def test_option_chain_is_usable_rejects_missing_column() -> None:
    df = pd.DataFrame([{"timestamp": "2024-01-01", "underlying": "SPY"}])
    assert option_chain_is_usable(df) is False


def test_option_chain_is_usable_none_and_empty() -> None:
    assert option_chain_is_usable(None) is False
    assert option_chain_is_usable(pd.DataFrame()) is False
