from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rlm.training.benchmarks import benchmark_coordinate_models


@dataclass
class ValidationSliceResult:
    symbol: str
    start: str
    end: str
    selected_realized_avg_improvement_vs_pr51: float
    top1_hit_rate_improvement_vs_pr51: float
    regime_flip_rate_improvement_vs_pr51: float
    drawdown_proxy_improvement_vs_pr51: float


def run_validation_matrix(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    windows: list[tuple[str, str]],
    horizon: int,
    sequence_window: int,
    smoothing_alpha: float,
    train_split: float = 0.8,
) -> list[ValidationSliceResult]:
    out: list[ValidationSliceResult] = []
    for symbol, df in frames_by_symbol.items():
        if "timestamp" not in df.columns:
            continue
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        for start, end in windows:
            mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end, tz="UTC"))
            sliced = df.loc[mask].reset_index(drop=True)
            if len(sliced) < max(100, horizon * 5):
                continue
            results = benchmark_coordinate_models(
                sliced,
                horizon=horizon,
                train_split=train_split,
                use_path_exits=True,
                sequence_window=sequence_window,
                smoothing_alpha=smoothing_alpha,
            )
            row_only = results["candidate_d_pr51_full"]
            temporal = results["candidate_f_pr53_temporal_full"]
            out.append(
                ValidationSliceResult(
                    symbol=symbol,
                    start=start,
                    end=end,
                    selected_realized_avg_improvement_vs_pr51=float(
                        temporal["selected_realized_average"] - row_only["selected_realized_average"]
                    ),
                    top1_hit_rate_improvement_vs_pr51=float(temporal["top1_hit_rate"] - row_only["top1_hit_rate"]),
                    regime_flip_rate_improvement_vs_pr51=float(
                        row_only["regime_flip_rate"] - temporal["regime_flip_rate"]
                    ),
                    drawdown_proxy_improvement_vs_pr51=float(temporal["drawdown_proxy"] - row_only["drawdown_proxy"]),
                )
            )
    return out


def summarize_validation_matrix(results: list[ValidationSliceResult]) -> dict[str, float]:
    if not results:
        return {
            "num_slices": 0.0,
            "win_rate_selected_realized_avg": 0.0,
            "win_rate_top1_hit": 0.0,
            "win_rate_flip_reduction": 0.0,
            "mean_selected_realized_avg_improvement": 0.0,
            "mean_flip_rate_improvement": 0.0,
        }

    n = len(results)
    return {
        "num_slices": float(n),
        "win_rate_selected_realized_avg": float(
            sum(r.selected_realized_avg_improvement_vs_pr51 > 0 for r in results) / n
        ),
        "win_rate_top1_hit": float(sum(r.top1_hit_rate_improvement_vs_pr51 > 0 for r in results) / n),
        "win_rate_flip_reduction": float(sum(r.regime_flip_rate_improvement_vs_pr51 > 0 for r in results) / n),
        "mean_selected_realized_avg_improvement": float(
            sum(r.selected_realized_avg_improvement_vs_pr51 for r in results) / n
        ),
        "mean_flip_rate_improvement": float(sum(r.regime_flip_rate_improvement_vs_pr51 for r in results) / n),
    }
