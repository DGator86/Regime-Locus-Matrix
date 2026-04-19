from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching
from rlm.roee.engine import ROEEConfig


def _bars(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = 500 + np.linspace(0, 10, n)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, n),
        },
        index=idx,
    )


def test_markov_vp_feature_vector_includes_optional_columns() -> None:
    df = _bars(20)
    df["vp_poc"] = df["close"] - 0.2
    df["vp_va_low"] = df["close"] - 1.0
    df["vp_va_high"] = df["close"] + 1.0
    df["cumulative_wyckoff_score"] = 0.8
    df["vp_hybrid_strength_max"] = 1.2
    df["vp_gex_confluence_poc"] = -0.3

    model = RLMMarkovSwitching(
        MarkovSwitchingConfig(
            use_intraday_vp_features=True,
            use_wyckoff_features=True,
            use_confluence_features=True,
        )
    )
    features = model._prepare_features(df)
    assert features is not None
    for col in (
        "vp_poc_distance",
        "vp_va_position",
        "cumulative_wyckoff_score",
        "vp_hybrid_strength_max",
        "vp_gex_confluence_poc",
    ):
        assert col in features.columns


def test_pipeline_collects_vp_signals_and_roee_uses_gating(monkeypatch) -> None:
    from rlm.core import pipeline as core_pipeline

    bars = _bars(120)

    def _mock_vp_compute(self, data: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["vp_poc"] = pd.to_numeric(data["close"], errors="coerce") - 0.2
        out["vp_va_high"] = pd.to_numeric(data["close"], errors="coerce") + 1.0
        out["vp_va_low"] = pd.to_numeric(data["close"], errors="coerce") - 1.0
        out["vp_hvn_count"] = 2
        out["vp_lvn_count"] = 2
        return out

    def _mock_wyckoff_compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"cumulative_wyckoff_score": [0.9] * len(data)}, index=data.index)

    def _mock_hybrid_compute(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "vp_gex_confluence_poc": [-0.5] * len(data),
                "vp_iv_skew_poc": [0.1] * len(data),
                "vp_hybrid_strength_max": [1.3] * len(data),
            },
            index=data.index,
        )

    monkeypatch.setattr(core_pipeline.MicrostructureVPFactors, "compute", _mock_vp_compute)
    monkeypatch.setattr(core_pipeline.CumulativeWyckoffFactors, "compute", _mock_wyckoff_compute)
    monkeypatch.setattr(core_pipeline.HybridConfluenceFactors, "compute", _mock_hybrid_compute)
    monkeypatch.setattr(
        core_pipeline,
        "hybrid_support_resistance",
        lambda *args, **kwargs: pd.DataFrame({"strength_score": [1.4]}),
    )

    cfg = FullRLMConfig(
        use_intraday_vp=True,
        use_cumulative_wyckoff=True,
        use_hybrid_confluence=True,
        roee_config=ROEEConfig(vp_gating_enabled=True),
    )
    result = FullRLMPipeline(cfg).run(bars)

    assert "vp_poc" in result.factors_df.columns
    assert "cumulative_wyckoff_score" in result.factors_df.columns
    assert result.vp_signals is not None
    assert "auction_state" in result.vp_signals
    assert "roee_action" in result.policy_df.columns
    assert not result.policy_df.empty
