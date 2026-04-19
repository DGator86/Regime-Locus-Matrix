from __future__ import annotations

import pandas as pd

from rlm.training.benchmarks import benchmark_coordinate_models


def _synthetic_df(n: int = 90) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": f"2026-03-01T00:{i:02d}:00Z",
                "symbol": "SPY",
                "close": 100 + (i * 0.25) + ((-1) ** (i // 5)) * 0.3,
                "sigma": 0.015 + (i % 6) * 0.001,
                "M_D": 5.0 + (i % 4),
                "M_V": 4.8 + (i % 3) * 0.4,
                "M_L": 5.1 + (i % 2) * 0.3,
                "M_G": 5.0 + (i % 5) * 0.2,
                "M_trend_strength": float((i % 5) - 2),
                "M_dealer_control": float(i % 4),
                "M_alignment": float((i % 3) - 1),
                "M_delta_neutral": float(i % 7),
                "M_R_trans": float(i % 3),
            }
        )
    return pd.DataFrame(rows)


def test_benchmark_runner_returns_expected_structure() -> None:
    out = benchmark_coordinate_models(_synthetic_df(), horizon=5, train_split=0.7)
    expected_models = {
        "baseline_a_bootstrap_fixed",
        "baseline_b_pr50",
        "candidate_c_pr51_targets",
        "candidate_d_pr51_full",
    }
    assert expected_models.issubset(out.keys())

    expected_metrics = {
        "selected_realized_average",
        "top1_hit_rate",
        "regime_accuracy_vs_teacher",
        "strategy_mse",
        "rank_correlation",
        "no_trade_frequency",
        "average_selected_edge",
        "drawdown_proxy",
    }
    for model_name in expected_models:
        assert expected_metrics.issubset(out[model_name].keys())
