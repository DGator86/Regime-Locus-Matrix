from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd

from rlm.factors.microstructure_vp_factors import MicrostructureVPFactors
from rlm.volume_profile.cumulative_wyckoff import cumulative_effort_result
from rlm.volume_profile.fx_session_profiles import get_fx_session_profile
from rlm.volume_profile.hybrid_confluence import vp_gex_confluence


def test_microstructure_vp_factors_computes_intraday_poc() -> None:
    idx = pd.date_range("2026-01-02 14:30:00", periods=4, freq="5s", tz="UTC")
    base = pd.DataFrame(index=idx)
    vp = pd.DataFrame(
        {
            "timestamp": idx,
            "vp_poc": [100.0, 100.2, 100.2, 100.4],
            "vp_va_high": [100.5, 100.6, 100.6, 100.8],
            "vp_va_low": [99.8, 99.9, 99.9, 100.0],
            "vp_hvn_count": [1, 2, 2, 3],
            "vp_lvn_count": [0, 1, 1, 1],
        }
    )

    with patch("rlm.factors.microstructure_vp_factors.rolling_intraday_vp", return_value=vp):
        out = MicrostructureVPFactors("SPY").compute(base)

    assert out["vp_poc"].iloc[-1] == 100.4
    assert out["vp_hvn_count"].iloc[-1] == 3


def test_cumulative_effort_result_divergence_patterns() -> None:
    bullish_absorption = pd.DataFrame(
        {
            "high": [101, 100.8, 100.6, 100.5],
            "low": [99.5, 99.7, 99.9, 100.0],
            "close": [100.2, 100.1, 100.0, 99.95],
            "volume": [100, 220, 350, 500],
        }
    )
    score = cumulative_effort_result(bullish_absorption)
    assert score > 0


def test_vp_gex_confluence_handles_missing_gex_data() -> None:
    with patch(
        "rlm.volume_profile.hybrid_confluence.MicrostructureDB.load_gex_surface",
        return_value=pd.DataFrame(),
    ):
        out = vp_gex_confluence("SPY", datetime(2026, 1, 2, 15, 0), [100.0, 101.0])

    assert out[100.0]["confluence_score"] == 0.0
    assert pd.isna(out[101.0]["net_gex"])


def test_get_fx_session_profile_filters_sydney_window() -> None:
    ts = pd.date_range("2026-01-02 20:00:00", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "price": [1.10, 1.11, 1.12, 1.12, 1.11, 1.10, 1.09, 1.08],
            "volume": [10, 20, 25, 15, 14, 12, 8, 7],
        }
    )

    profile = get_fx_session_profile(df, "Sydney", datetime(2026, 1, 2, 22, 0))
    assert not pd.isna(profile["poc"])
