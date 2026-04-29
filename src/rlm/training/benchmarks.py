from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.features.scoring.coordinate_regime_bootstrap import bootstrap_regime_label_from_coordinates
from rlm.features.scoring.regime_model_temporal import TemporalRegimeModel
from rlm.features.scoring.regime_smoother import regime_flip_rate
from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.roee.strategy_value_model_temporal import TemporalStrategyValueModel
from rlm.training.datasets import (
    REQUIRED_COORD_COLUMNS,
    build_regime_training_frame,
    build_strategy_value_training_frame,
)
from rlm.training.train_coordinate_models import (
    compute_strategy_metrics,
    train_regime_model,
    train_strategy_value_model,
)


def benchmark_coordinate_models(
    df: pd.DataFrame,
    horizon: int,
    train_split: float = 0.8,
    *,
    use_path_exits: bool = True,
    sequence_window: int | None = None,
    smoothing_alpha: float = 0.25,
) -> dict[str, dict[str, float | dict[str, float]]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0,1)")

    eval_frame = build_strategy_value_training_frame(
        df,
        horizon=horizon,
        target_mode="v2",
        use_path_exits=use_path_exits,
    )
    if len(eval_frame) < 5:
        raise ValueError("Not enough rows for benchmark")

    cut = int(len(eval_frame) * train_split)
    train_base = df.iloc[: cut + horizon].reset_index(drop=True)
    val_eval = eval_frame.iloc[cut:].reset_index(drop=True)
    x_val = val_eval.loc[:, REQUIRED_COORD_COLUMNS]
    y_val = val_eval.loc[:, StrategyValueModel().strategies].to_numpy(dtype=float)
    teacher_labels = [bootstrap_regime_label_from_coordinates(r) for r in x_val.to_dict(orient="records")]

    results: dict[str, dict[str, float | dict[str, float]]] = {}

    def evaluate(name: str, regime_model, value_model, *, temporal: bool = False) -> None:
        pred = value_model.predict_expected_values(x_val)
        pred_best = pred.argmax(axis=1)
        selected = y_val[np.arange(len(y_val)), pred_best]
        strategy_metrics = compute_strategy_metrics(value_model, val_eval)

        regime_pred = regime_model.predict(x_val)
        regime_acc_teacher = float(np.mean(np.array(regime_pred) == np.array(teacher_labels)))
        no_trade_idx = value_model.strategies.index("no_trade")
        results[name] = {
            "selected_realized_average": float(np.mean(selected)) if len(selected) else 0.0,
            "top1_hit_rate": float(strategy_metrics.top1_hit_rate),
            "regime_accuracy_vs_teacher": regime_acc_teacher,
            "strategy_mse": float(np.mean(list(strategy_metrics.mse_per_strategy.values()))),
            "rank_correlation": float(strategy_metrics.rank_correlation),
            "no_trade_frequency": (float(np.mean(pred_best == no_trade_idx)) if len(pred_best) else 0.0),
            "average_selected_edge": (float(np.mean(selected - y_val.mean(axis=1))) if len(selected) else 0.0),
            "drawdown_proxy": float(np.min(np.cumsum(selected))) if len(selected) else 0.0,
            "strategy_mse_per_strategy": strategy_metrics.mse_per_strategy,
            "regime_flip_rate": float(regime_flip_rate(regime_pred)),
            "smoothed_no_trade_frequency": (
                float(np.mean(pred_best == no_trade_idx)) if temporal and len(pred_best) else 0.0
            ),
        }

    baseline_regime = train_regime_model(build_regime_training_frame(train_base, label_mode="bootstrap"))
    evaluate(
        "baseline_a_bootstrap_fixed",
        baseline_regime,
        StrategyValueModel.with_bootstrap_coefficients(),
    )

    b_regime = train_regime_model(build_regime_training_frame(train_base, label_mode="bootstrap"))
    b_value = train_strategy_value_model(
        build_strategy_value_training_frame(train_base, horizon=horizon, target_mode="v1")
    )
    evaluate("baseline_b_pr50", b_regime, b_value)

    c_regime = train_regime_model(build_regime_training_frame(train_base, label_mode="bootstrap"))
    c_value = train_strategy_value_model(
        build_strategy_value_training_frame(
            train_base,
            horizon=horizon,
            target_mode="v2",
            use_path_exits=use_path_exits,
        )
    )
    evaluate("candidate_c_pr51_targets", c_regime, c_value)

    d_regime = train_regime_model(build_regime_training_frame(train_base, label_mode="outcome", horizon=horizon))
    d_value = train_strategy_value_model(
        build_strategy_value_training_frame(
            train_base,
            horizon=horizon,
            target_mode="v2",
            use_path_exits=use_path_exits,
        )
    )
    evaluate("candidate_d_pr51_full", d_regime, d_value)

    if sequence_window is not None:
        e_regime = train_regime_model(
            build_regime_training_frame(
                train_base,
                label_mode="bootstrap",
                sequence_window=sequence_window,
            )
        )
        e_value = train_strategy_value_model(
            build_strategy_value_training_frame(
                train_base,
                horizon=horizon,
                target_mode="v2",
                use_path_exits=use_path_exits,
                sequence_window=sequence_window,
            )
        )
        evaluate(
            "candidate_e_pr53_temporal_bootstrap_labels",
            TemporalRegimeModel(e_regime, window=sequence_window, smoothing_alpha=smoothing_alpha),
            TemporalStrategyValueModel(e_value, window=sequence_window),
            temporal=True,
        )

        f_regime = train_regime_model(
            build_regime_training_frame(
                train_base,
                label_mode="outcome",
                horizon=horizon,
                sequence_window=sequence_window,
            )
        )
        f_value = train_strategy_value_model(
            build_strategy_value_training_frame(
                train_base,
                horizon=horizon,
                target_mode="v2",
                use_path_exits=use_path_exits,
                sequence_window=sequence_window,
            )
        )
        evaluate(
            "candidate_f_pr53_temporal_full",
            TemporalRegimeModel(f_regime, window=sequence_window, smoothing_alpha=smoothing_alpha),
            TemporalStrategyValueModel(f_value, window=sequence_window),
            temporal=True,
        )

    return results


def summarize_benchmark_results(
    results: dict[str, dict[str, float | dict[str, float]]],
) -> dict[str, float]:
    baseline = results["baseline_b_pr50"]
    candidate = results["candidate_d_pr51_full"]
    if "candidate_f_pr53_temporal_full" not in results:
        return {
            "selected_realized_avg_improvement": float(
                candidate["selected_realized_average"] - baseline["selected_realized_average"]
            ),
            "top1_hit_rate_improvement": float(candidate["top1_hit_rate"] - baseline["top1_hit_rate"]),
            "rank_correlation_improvement": float(candidate["rank_correlation"] - baseline["rank_correlation"]),
            "drawdown_proxy_improvement": float(candidate["drawdown_proxy"] - baseline["drawdown_proxy"]),
        }

    row_only = results["candidate_d_pr51_full"]
    temporal = results["candidate_f_pr53_temporal_full"]
    return {
        "selected_realized_avg_improvement_vs_pr51": float(
            temporal["selected_realized_average"] - row_only["selected_realized_average"]
        ),
        "top1_hit_rate_improvement_vs_pr51": float(temporal["top1_hit_rate"] - row_only["top1_hit_rate"]),
        "regime_flip_rate_improvement_vs_pr51": float(
            row_only.get("regime_flip_rate", 0.0) - temporal.get("regime_flip_rate", 0.0)
        ),
        "drawdown_proxy_improvement_vs_pr51": float(temporal["drawdown_proxy"] - row_only["drawdown_proxy"]),
    }
