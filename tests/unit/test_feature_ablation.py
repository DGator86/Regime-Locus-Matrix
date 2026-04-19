from __future__ import annotations

import pandas as pd

from rlm.training.feature_ablation import run_temporal_ablation


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    for i in range(n):
        rows.append(
            {
                "timestamp": (start + pd.Timedelta(minutes=i)).isoformat(),
                "symbol": "SPY",
                "close": 100 + i * 0.1,
                "sigma": 0.02,
                "M_D": 5.0 + (i % 4),
                "M_V": 4.8 + (i % 3) * 0.4,
                "M_L": 5.0 + (i % 2) * 0.5,
                "M_G": 5.0 + (i % 5) * 0.2,
                "M_trend_strength": float(i % 4),
                "M_dealer_control": float(i % 3),
                "M_alignment": float((i % 5) - 2),
                "M_delta_neutral": float(i % 6),
                "M_R_trans": float(i % 3),
            }
        )
    return pd.DataFrame(rows)


def test_ablation_returns_full_and_group_drops() -> None:
    out = run_temporal_ablation(
        _synthetic_df(),
        horizon=5,
        sequence_window=5,
        smoothing_alpha=0.25,
    )
    assert "full" in out
    assert "drop_means" in out
    assert "drop_stds" in out
    assert "drop_deltas" in out
    assert "drop_slopes" in out
