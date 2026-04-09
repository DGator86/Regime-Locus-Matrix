"""Tests for option-chain → bar enrichment (dealer / liquidity fields)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlm.datasets import bars_enrichment
from rlm.datasets.bars_enrichment import (
    enrich_bars_from_option_chain,
    enrich_bars_with_vix,
    prepare_bars_for_factors,
)


def _minimal_chain(ts: pd.Timestamp, spot: float) -> pd.DataFrame:
    exp = ts + pd.Timedelta(days=30)
    rows = []
    for opt, strike_off in [("call", 0), ("call", 5), ("put", 0), ("put", -5)]:
        k = round((spot + strike_off) / 5) * 5
        rows.append(
            {
                "timestamp": ts,
                "underlying": "SPY",
                "expiry": exp,
                "option_type": opt,
                "strike": float(k),
                "bid": 1.0,
                "ask": 1.1,
                "iv": 0.22,
                "gamma": 0.02,
                "delta": 0.5 if opt == "call" else -0.5,
                "vanna": 0.01,
                "charm": -0.02,
                "open_interest": 5000.0,
                "volume": 800.0,
            }
        )
    return pd.DataFrame(rows)


def test_enrich_bars_from_option_chain_adds_dealer_columns() -> None:
    idx = pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"])
    bars = pd.DataFrame(
        {
            "open": [500.0, 501.0, 502.0],
            "high": [502.0, 503.0, 504.0],
            "low": [499.0, 500.0, 501.0],
            "close": [501.0, 502.0, 503.0],
            "volume": [1e6, 1e6, 1e6],
            "vwap": [500.5, 501.5, 502.5],
        },
        index=idx,
    )
    bars.index.name = "timestamp"

    c1 = _minimal_chain(idx[0], 501.0)
    c2 = _minimal_chain(idx[1], 502.0)
    chain = pd.concat([c1, c2], ignore_index=True)

    out = enrich_bars_from_option_chain(bars, chain, underlying="SPY")
    assert np.isfinite(out.loc[idx[0], "gex"])
    assert np.isfinite(out.loc[idx[1], "gex"])
    assert "bid_ask_spread" in out.columns
    assert "options_spread_pct_mid" in out.columns
    assert "options_volume" in out.columns
    assert "options_volume_to_oi" in out.columns
    assert np.isfinite(out.loc[idx[0], "options_spread_pct_mid"])
    assert np.isfinite(out.loc[idx[0], "options_volume"])
    assert np.isfinite(out.loc[idx[0], "options_volume_to_oi"])
    assert out.loc[idx[2], "gex"] == 0.0  # neutral fill without chain row


def test_prepare_bars_skips_vix_when_disabled() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="D")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
            "vwap": 100.0,
        },
        index=idx,
    )
    out = prepare_bars_for_factors(bars, None, underlying="SPY", attach_vix=False)
    assert "vix" not in out.columns or out["vix"].isna().all()


def test_enrich_bars_with_vix_uses_last_known_intraday_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.to_datetime(
        [
            "2026-04-07 09:30:00",
            "2026-04-07 09:31:00",
            "2026-04-07 09:33:00",
            "2026-04-07 09:36:00",
        ]
    )
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
            "vwap": 100.0,
        },
        index=idx,
    )
    bars.index.name = "timestamp"

    vix_src = pd.Series(
        [20.0, 20.4, 20.8],
        index=pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2026-04-07 09:31:00",
                    "2026-04-07 09:33:00",
                    "2026-04-07 09:35:00",
                ]
            )
        ).tz_localize("America/New_York"),
        dtype=float,
    )
    vvix_src = pd.Series(
        [100.0, 101.5],
        index=pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2026-04-07 09:32:00",
                    "2026-04-07 09:34:00",
                ]
            )
        ).tz_localize("America/New_York"),
        dtype=float,
    )

    def _fake_loader(sym: str, bars_index: pd.Index) -> pd.Series:
        bars_ts = bars_enrichment._to_exchange_time(bars_index)
        source = vix_src if sym == "^VIX" else vvix_src
        out = pd.Series(np.nan, index=bars_ts, dtype=float)
        out.loc[:] = source.reindex(bars_ts, method="ffill").to_numpy()
        return out

    monkeypatch.setattr(bars_enrichment, "_load_intraday_vix_series", _fake_loader)

    out = enrich_bars_with_vix(bars)

    assert np.isnan(out.loc[idx[0], "vix"])
    assert out.loc[idx[1], "vix"] == 20.0
    assert out.loc[idx[2], "vix"] == 20.4
    assert out.loc[idx[3], "vix"] == 20.8

    assert np.isnan(out.loc[idx[0], "vvix"])
    assert np.isnan(out.loc[idx[1], "vvix"])
    assert out.loc[idx[2], "vvix"] == 100.0
    assert out.loc[idx[3], "vvix"] == 101.5


def test_enrich_bars_with_vix_does_not_carry_intraday_values_across_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.to_datetime(
        [
            "2026-04-07 15:59:00",
            "2026-04-08 09:30:00",
            "2026-04-08 09:31:00",
        ]
    )
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1e6,
            "vwap": 100.0,
        },
        index=idx,
    )
    bars.index.name = "timestamp"

    source = pd.Series(
        [22.0, 19.5],
        index=pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2026-04-07 15:59:00",
                    "2026-04-08 09:31:00",
                ]
            )
        ).tz_localize("America/New_York"),
        dtype=float,
    )

    def _fake_loader(sym: str, bars_index: pd.Index) -> pd.Series:
        bars_ts = bars_enrichment._to_exchange_time(bars_index)
        out = pd.Series(np.nan, index=bars_ts, dtype=float)
        for day in pd.Index(bars_ts.normalize().unique()).sort_values():
            day_end = day + pd.Timedelta(days=1)
            day_bars = bars_ts[(bars_ts >= day) & (bars_ts < day_end)]
            day_source = source[(source.index >= day) & (source.index < day_end)]
            if day_source.empty:
                continue
            out.loc[day_bars] = day_source.reindex(day_bars, method="ffill").to_numpy()
        return out

    monkeypatch.setattr(bars_enrichment, "_load_intraday_vix_series", _fake_loader)

    out = enrich_bars_with_vix(bars)

    assert out.loc[idx[0], "vix"] == 22.0
    assert np.isnan(out.loc[idx[1], "vix"])
    assert out.loc[idx[2], "vix"] == 19.5
