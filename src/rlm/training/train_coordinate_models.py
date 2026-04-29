from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rlm.features.scoring.regime_model import REGIME_LABELS, RegimeModel
from rlm.roee.strategy_value_model import STRATEGY_NAMES, StrategyValueModel
from rlm.training.artifacts import (
    RegimeModelArtifact,
    StrategyValueModelArtifact,
    load_regime_model_artifact,
    load_strategy_value_model_artifact,
    save_artifact,
)
from rlm.training.datasets import REQUIRED_COORD_COLUMNS

ARTIFACT_DIR = Path("artifacts/models")
REGIME_ARTIFACT_PATH = ARTIFACT_DIR / "regime_model.json"
STRATEGY_VALUE_ARTIFACT_PATH = ARTIFACT_DIR / "strategy_value_model.json"

REGIME_FEATURE_NAMES: list[str] = [
    "intercept",
    "d",
    "v",
    "liquidity",
    "g",
    "trend",
    "dealer",
    "align",
    "delta",
    "r_trans",
    "d*g",
    "v*liquidity",
    "align^2",
    "r_trans^2",
    "liquidity^2",
]

STRATEGY_FEATURE_NAMES: list[str] = [
    "intercept",
    "d",
    "v",
    "liquidity",
    "g",
    "trend",
    "align",
    "delta",
    "r_trans",
    "d*g",
    "v*liquidity",
    "align^2",
    "r_trans^2",
]


@dataclass
class RegimeMetrics:
    training_accuracy: float
    validation_accuracy: float
    log_loss: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]


@dataclass
class StrategyMetrics:
    mse_per_strategy: dict[str, float]
    rank_correlation: float
    top1_hit_rate: float
    selected_realized_average: float
    bootstrap_selected_average: float


def train_regime_model(train_df: pd.DataFrame) -> RegimeModel:
    X = train_df.loc[:, REQUIRED_COORD_COLUMNS]
    y = train_df["regime_label"].astype(str).tolist()
    return RegimeModel().fit(X, y)


def train_strategy_value_model(train_df: pd.DataFrame) -> StrategyValueModel:
    X = train_df.loc[:, REQUIRED_COORD_COLUMNS]
    Y = train_df.loc[:, STRATEGY_NAMES]
    return StrategyValueModel().fit(X, Y)


def save_model_artifacts(
    regime_model: RegimeModel,
    value_model: StrategyValueModel,
    *,
    regime_training_rows: int,
    value_training_rows: int,
    source_symbols: list[str],
    out_dir: str | Path = ARTIFACT_DIR,
    target_mode: str = "unknown",
    label_mode: str = "unknown",
    horizon: int = 0,
    training_start: str | None = None,
    training_end: str | None = None,
    benchmark_summary: dict[str, float] | None = None,
    simulator_version: str | None = None,
    execution_model_version: str | None = None,
    train_split: float | None = None,
    validation_rows: int | None = None,
    sequence_window: int | None = None,
    smoothing_alpha: float | None = None,
    temporal_model: bool = False,
    validation_matrix_summary: dict[str, float] | None = None,
    feature_ablation_summary: dict[str, float] | None = None,
    model_health_snapshot: dict[str, float | bool] | None = None,
    refresh_parent_version: str | None = None,
    refresh_reason: str | None = None,
    promotion_status: str | None = None,
) -> tuple[Path, Path]:
    out_dir_path = Path(out_dir)
    trained_at = datetime.now(UTC).isoformat()

    regime_artifact = regime_model.to_artifact(
        trained_at=trained_at,
        training_rows=regime_training_rows,
        source_symbols=source_symbols,
        feature_names=REGIME_FEATURE_NAMES,
        target_mode=target_mode,
        label_mode=label_mode,
        horizon=horizon,
        training_start=training_start,
        training_end=training_end,
        benchmark_summary=benchmark_summary,
        simulator_version=simulator_version,
        execution_model_version=execution_model_version,
        train_split=train_split,
        validation_rows=validation_rows,
        sequence_window=sequence_window,
        smoothing_alpha=smoothing_alpha,
        temporal_model=temporal_model,
        validation_matrix_summary=validation_matrix_summary,
        feature_ablation_summary=feature_ablation_summary,
        model_health_snapshot=model_health_snapshot,
        refresh_parent_version=refresh_parent_version,
        refresh_reason=refresh_reason,
        promotion_status=promotion_status,
    )
    value_artifact = value_model.to_artifact(
        trained_at=trained_at,
        training_rows=value_training_rows,
        source_symbols=source_symbols,
        feature_names=STRATEGY_FEATURE_NAMES,
        target_mode=target_mode,
        label_mode=label_mode,
        horizon=horizon,
        training_start=training_start,
        training_end=training_end,
        benchmark_summary=benchmark_summary,
        simulator_version=simulator_version,
        execution_model_version=execution_model_version,
        train_split=train_split,
        validation_rows=validation_rows,
        sequence_window=sequence_window,
        smoothing_alpha=smoothing_alpha,
        temporal_model=temporal_model,
        validation_matrix_summary=validation_matrix_summary,
        feature_ablation_summary=feature_ablation_summary,
        model_health_snapshot=model_health_snapshot,
        refresh_parent_version=refresh_parent_version,
        refresh_reason=refresh_reason,
        promotion_status=promotion_status,
    )

    regime_path = out_dir_path / REGIME_ARTIFACT_PATH.name
    value_path = out_dir_path / STRATEGY_VALUE_ARTIFACT_PATH.name
    save_artifact(regime_path, regime_artifact)
    save_artifact(value_path, value_artifact)
    return regime_path, value_path


def load_model_artifacts(
    out_dir: str | Path = ARTIFACT_DIR,
) -> tuple[RegimeModelArtifact, StrategyValueModelArtifact]:
    out_dir_path = Path(out_dir)
    return (
        load_regime_model_artifact(out_dir_path / REGIME_ARTIFACT_PATH.name),
        load_strategy_value_model_artifact(out_dir_path / STRATEGY_VALUE_ARTIFACT_PATH.name),
    )


def load_trained_regime_model_or_bootstrap(path: str | Path = REGIME_ARTIFACT_PATH) -> RegimeModel:
    artifact_path = Path(path)
    if artifact_path.exists():
        return RegimeModel.from_artifact(load_regime_model_artifact(artifact_path))
    return RegimeModel.with_bootstrap_coefficients()


def load_trained_value_model_or_bootstrap(
    path: str | Path = STRATEGY_VALUE_ARTIFACT_PATH,
) -> StrategyValueModel:
    artifact_path = Path(path)
    if artifact_path.exists():
        return StrategyValueModel.from_artifact(load_strategy_value_model_artifact(artifact_path))
    return StrategyValueModel.with_bootstrap_coefficients()


def compute_regime_metrics(
    model: RegimeModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> RegimeMetrics:
    train_y = train_df["regime_label"].astype(str).to_numpy()
    val_y = val_df["regime_label"].astype(str).to_numpy()
    train_pred = np.array(model.predict(train_df.loc[:, REQUIRED_COORD_COLUMNS]))
    val_pred = np.array(model.predict(val_df.loc[:, REQUIRED_COORD_COLUMNS]))
    val_probs = model.predict_proba(val_df.loc[:, REQUIRED_COORD_COLUMNS])
    label_to_idx = {label: i for i, label in enumerate(REGIME_LABELS)}

    ll = 0.0
    for i, label in enumerate(val_y):
        ll += -np.log(max(val_probs[i, label_to_idx[label]], 1e-12))
    log_loss = ll / max(len(val_y), 1)

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for label in REGIME_LABELS:
        tp = float(np.sum((val_pred == label) & (val_y == label)))
        fp = float(np.sum((val_pred == label) & (val_y != label)))
        fn = float(np.sum((val_pred != label) & (val_y == label)))
        precision[label] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[label] = tp / (tp + fn) if tp + fn > 0 else 0.0

    return RegimeMetrics(
        training_accuracy=float(np.mean(train_pred == train_y)) if len(train_y) else 0.0,
        validation_accuracy=float(np.mean(val_pred == val_y)) if len(val_y) else 0.0,
        log_loss=float(log_loss),
        per_class_precision=precision,
        per_class_recall=recall,
    )


def compute_strategy_metrics(
    model: StrategyValueModel,
    val_df: pd.DataFrame,
    baseline_model: StrategyValueModel | None = None,
) -> StrategyMetrics:
    X = val_df.loc[:, REQUIRED_COORD_COLUMNS]
    Y = val_df.loc[:, STRATEGY_NAMES].to_numpy(dtype=float)
    pred = model.predict_expected_values(X)
    mse = {name: float(np.mean((pred[:, i] - Y[:, i]) ** 2)) for i, name in enumerate(STRATEGY_NAMES)}

    predicted_best = pred.argmax(axis=1)
    realized_best = Y.argmax(axis=1)
    top1 = float(np.mean(predicted_best == realized_best)) if len(Y) else 0.0

    selected_values = Y[np.arange(len(Y)), predicted_best] if len(Y) else np.array([], dtype=float)
    selected_avg = float(np.mean(selected_values)) if len(selected_values) else 0.0

    baseline = baseline_model or StrategyValueModel.with_bootstrap_coefficients()
    base_pred = baseline.predict_expected_values(X)
    base_best = base_pred.argmax(axis=1)
    base_selected = Y[np.arange(len(Y)), base_best] if len(Y) else np.array([], dtype=float)
    base_avg = float(np.mean(base_selected)) if len(base_selected) else 0.0

    pred_rank = np.argsort(np.argsort(pred, axis=1), axis=1)
    real_rank = np.argsort(np.argsort(Y, axis=1), axis=1)
    if len(Y):
        corr = float(np.mean([np.corrcoef(pred_rank[i], real_rank[i])[0, 1] for i in range(len(Y))]))
    else:
        corr = 0.0

    return StrategyMetrics(
        mse_per_strategy=mse,
        rank_correlation=corr,
        top1_hit_rate=top1,
        selected_realized_average=selected_avg,
        bootstrap_selected_average=base_avg,
    )
