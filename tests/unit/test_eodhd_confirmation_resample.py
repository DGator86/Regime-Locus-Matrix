"""EODHD lake must honor requested confirmation bar sizes (not raw 1m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlm.data.bar_timeframes import pandas_resample_rule_for_bar_size
from rlm.data.stock_bars_provider import fetch_stock_bars, resample_ohlcv_bars


def _synthetic_1m(n: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-07-01 09:30", periods=n, freq="min")
    close = 100 + np.cumsum(rng.normal(0, 0.01, size=n))
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": rng.integers(100, 500, size=n),
            "vwap": close,
        }
    )


def test_pandas_resample_rule_for_confirmation_sizes() -> None:
    assert pandas_resample_rule_for_bar_size("1 min") is None
    assert pandas_resample_rule_for_bar_size("5 mins") == "5min"
    assert pandas_resample_rule_for_bar_size("15 mins") == "15min"


def test_resample_ohlcv_bars_aggregates_1m_to_5m() -> None:
    idx = pd.date_range("2026-07-01 09:30", periods=15, freq="min")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": range(15),
            "high": range(15, 30),
            "low": range(15),
            "close": range(100, 115),
            "volume": [1] * 15,
            "vwap": range(15),
        }
    )
    out = resample_ohlcv_bars(df, "5 mins")
    assert len(out) == 3
    assert float(out.iloc[0]["open"]) == 0.0
    assert float(out.iloc[0]["close"]) == 104.0
    assert float(out.iloc[0]["volume"]) == 5.0


def _rth_1m_across_days(n_days: int = 5) -> pd.DataFrame:
    """RTH-only 1m prints (no overnight rows) spanning a weekend gap."""
    frames: list[pd.DataFrame] = []
    px = 100.0
    # Thu Jul 2 .. Wed Jul 8 2026 includes a weekend
    for day in pd.bdate_range("2026-07-02", periods=n_days):
        idx = pd.date_range(f"{day.date()} 09:30", periods=60, freq="min")
        close = px + np.arange(len(idx)) * 0.01
        px = float(close[-1]) + 0.5
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "open": close,
                    "high": close + 0.02,
                    "low": close - 0.02,
                    "close": close,
                    "volume": np.full(len(idx), 10.0),
                    "vwap": close,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_resample_ohlcv_bars_drops_overnight_empty_bins() -> None:
    """5m/15m primary must not keep overnight NaN OHLC + volume=0 ghost bars."""
    lake = _rth_1m_across_days(5)
    out = resample_ohlcv_bars(lake, "5 mins")
    assert not out.empty
    assert int(out["close"].isna().sum()) == 0
    # 60 RTH minutes / day → 12 five-minute bars; no overnight expansion
    assert len(out) == 5 * 12
    hours = pd.to_datetime(out["timestamp"]).dt.hour
    assert hours.min() >= 9
    assert hours.max() <= 10  # 09:30–10:29 stubs


def test_fetch_stock_bars_daily_drops_weekend_empty_bins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three-track daily primary (EODHD lake→1D) must not inject Sat/Sun NaN rows."""
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)
    lake = _rth_1m_across_days(5)
    monkeypatch.setattr(
        "rlm.data.stock_bars_provider.load_stock_1m_from_lake",
        lambda *a, **k: lake.copy(),
    )
    daily = fetch_stock_bars("SPY", duration="30 D", bar_size="1 day", data_root=tmp_path)
    assert len(daily) == 5
    assert int(daily["close"].isna().sum()) == 0
    assert int(pd.to_datetime(daily["timestamp"]).dt.weekday.ge(5).sum()) == 0


def test_fetch_stock_bars_eodhd_resamples_confirmation_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLM_STOCK_BARS_SOURCE", "eodhd")
    monkeypatch.setenv("RLM_EODHD_MIN_LAKE_BARS", "100")
    monkeypatch.delenv("RLM_ALLOW_IBKR_STOCK_BARS", raising=False)
    lake = _synthetic_1m(2_000)
    monkeypatch.setattr(
        "rlm.data.stock_bars_provider.load_stock_1m_from_lake",
        lambda *a, **k: lake.copy(),
    )

    raw_1m = fetch_stock_bars(
        "SPY", duration="10 D", bar_size="1 min", data_root=tmp_path
    )
    conf_5m = fetch_stock_bars(
        "SPY", duration="10 D", bar_size="5 mins", data_root=tmp_path
    )
    assert len(raw_1m) >= 100
    assert len(conf_5m) < len(raw_1m)
    # ~5× fewer bars when aggregating 1m → 5m (allow gaps at edges)
    assert len(conf_5m) <= max(1, len(raw_1m) // 4)


def test_direction_aligned_nan_is_inconclusive() -> None:
    from scripts.run_universe_options_pipeline import _direction_aligned, _finite_sd

    assert _finite_sd(float("nan")) is None
    assert _direction_aligned(None, 0.4) is True
    assert _direction_aligned(0.4, None) is True
    assert _direction_aligned(0.4, -0.2) is False
    assert _direction_aligned(0.4, 0.1) is True
    assert _direction_aligned(0.0, 0.3) is False
