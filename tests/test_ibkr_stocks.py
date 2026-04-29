"""Unit tests for IBKR stock bar helpers (no live TWS connection)."""

from __future__ import annotations

import builtins
import importlib
from typing import Any

import pandas as pd
import pytest

from rlm.data import ibkr_stocks


def test_load_ibkr_socket_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ibkr_stocks, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("IBKR_HOST", raising=False)
    monkeypatch.delenv("IBKR_PORT", raising=False)
    monkeypatch.delenv("IBKR_CLIENT_ID", raising=False)
    h, p, cid = ibkr_stocks.load_ibkr_socket_config()
    assert h == "127.0.0.1"
    assert p == 7497
    assert cid == 1


def test_load_ibkr_socket_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5")
    monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_CLIENT_ID", "7")
    h, p, cid = ibkr_stocks.load_ibkr_socket_config()
    assert h == "10.0.0.5"
    assert p == 4002
    assert cid == 7


def test_ibkr_bars_to_dataframe_empty() -> None:
    df = ibkr_stocks.ibkr_bars_to_dataframe([])
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
    assert len(df) == 0


def test_ibkr_bars_to_dataframe_from_dicts() -> None:
    bars = [
        {
            "date": "20240102",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
            "wap": 100.25,
        },
        {
            "date": "20240103",
            "open": 100.5,
            "high": 102.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 900_000,
            "wap": 101.1,
        },
    ]
    df = ibkr_stocks.ibkr_bars_to_dataframe(bars)
    assert len(df) == 2
    assert df["close"].iloc[-1] == 101.0
    assert df["volume"].iloc[0] == 1_000_000.0
    assert not pd.isna(df["vwap"].iloc[0])


def test_ibkr_bars_to_dataframe_uses_average_when_no_wap() -> None:
    """TWS historical ``BarData`` populates ``average``, not ``wap``."""
    bars = [
        {
            "date": "20240102",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
            "average": 100.33,
        },
    ]
    df = ibkr_stocks.ibkr_bars_to_dataframe(bars)
    assert df["vwap"].iloc[0] == pytest.approx(100.33)


def test_fetch_historical_stock_bars_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ibapi, fetch should raise a clear ImportError."""
    real_import = builtins.__import__

    def block_ibapi(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ibapi" or str(name).startswith("ibapi."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_ibapi)
    importlib.reload(ibkr_stocks)
    try:
        with pytest.raises(ImportError, match=r"regime-locus-matrix\[ibkr\]"):
            ibkr_stocks.fetch_historical_stock_bars("SPY", timeout_sec=1.0)
    finally:
        importlib.reload(ibkr_stocks)
