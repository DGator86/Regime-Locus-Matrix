"""Tests for option-chain → bar enrichment (dealer / liquidity fields)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.datasets.bars_enrichment import enrich_bars_from_option_chain, prepare_bars_for_factors


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
