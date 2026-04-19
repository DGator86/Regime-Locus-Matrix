from __future__ import annotations

import pandas as pd

from rlm.training.validation_matrix import run_validation_matrix, summarize_validation_matrix


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


def test_validation_matrix_returns_slice_results() -> None:
    frames = {"SPY": _synthetic_df(200)}
    windows = [("2026-03-01T00:00:00Z", "2026-03-01T02:59:00Z")]
    out = run_validation_matrix(
        frames,
        windows=windows,
        horizon=5,
        sequence_window=5,
        smoothing_alpha=0.25,
    )
    assert len(out) >= 1


def test_validation_summary_has_expected_keys() -> None:
    frames = {"SPY": _synthetic_df(200)}
    windows = [("2026-03-01T00:00:00Z", "2026-03-01T02:59:00Z")]
    summary = summarize_validation_matrix(
        run_validation_matrix(
            frames,
            windows=windows,
            horizon=5,
            sequence_window=5,
            smoothing_alpha=0.25,
        )
    )
    assert set(summary) == {
        "num_slices",
        "win_rate_selected_realized_avg",
        "win_rate_top1_hit",
        "win_rate_flip_reduction",
        "mean_selected_realized_avg_improvement",
        "mean_flip_rate_improvement",
    }
