from __future__ import annotations

import pandas as pd

from rlm.training.benchmarks import benchmark_coordinate_models
from rlm.training.datasets import REQUIRED_COORD_COLUMNS

ABLATION_GROUPS = {
    "means": ["_mean_"],
    "stds": ["_std_"],
    "deltas": ["_delta_"],
    "slopes": ["_slope_"],
}


def drop_feature_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if group_name not in ABLATION_GROUPS:
        raise ValueError(f"Unknown group: {group_name}")
    pats = ABLATION_GROUPS[group_name]
    keep = [
        c
        for c in df.columns
        if c in REQUIRED_COORD_COLUMNS or not any(p in c for p in pats)
    ]
    return df.loc[:, keep].copy()


def run_temporal_ablation(
    df: pd.DataFrame,
    *,
    horizon: int,
    sequence_window: int,
    smoothing_alpha: float,
    train_split: float = 0.8,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    full = benchmark_coordinate_models(
        df,
        horizon=horizon,
        train_split=train_split,
        use_path_exits=True,
        sequence_window=sequence_window,
        smoothing_alpha=smoothing_alpha,
    )
    results["full"] = full["candidate_f_pr53_temporal_full"]

    for group in ABLATION_GROUPS:
        pruned = drop_feature_group(df, group)
        res = benchmark_coordinate_models(
            pruned,
            horizon=horizon,
            train_split=train_split,
            use_path_exits=True,
            sequence_window=sequence_window,
            smoothing_alpha=smoothing_alpha,
        )
        results[f"drop_{group}"] = res["candidate_f_pr53_temporal_full"]

    return results


def summarize_ablation(results: dict[str, dict]) -> dict[str, float]:
    base = results["full"]
    out: dict[str, float] = {}
    for name, metrics in results.items():
        if name == "full":
            continue
        out[f"{name}_selected_realized_delta"] = float(
            metrics["selected_realized_average"] - base["selected_realized_average"]
        )
        out[f"{name}_flip_rate_delta"] = float(
            metrics["regime_flip_rate"] - base["regime_flip_rate"]
        )
    return out
