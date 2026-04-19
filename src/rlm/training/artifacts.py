from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RegimeModelArtifact:
    labels: list[str]
    weights: list[list[float]]
    feature_names: list[str]
    trained_at: str
    training_rows: int
    source_symbols: list[str]
    target_mode: str | None = None
    label_mode: str | None = None
    horizon: int | None = None
    training_start: str | None = None
    training_end: str | None = None
    benchmark_summary: dict[str, float] | None = None
    simulator_version: str | None = None
    execution_model_version: str | None = None
    train_split: float | None = None
    validation_rows: int | None = None
    sequence_window: int | None = None
    smoothing_alpha: float | None = None
    temporal_model: bool = False
    validation_matrix_summary: dict[str, float] | None = None
    feature_ablation_summary: dict[str, float] | None = None
    model_health_snapshot: dict[str, float | bool] | None = None
    refresh_parent_version: str | None = None
    refresh_reason: str | None = None
    promotion_status: str | None = None


@dataclass
class StrategyValueModelArtifact:
    strategies: list[str]
    coefficients: list[list[float]]
    feature_names: list[str]
    trained_at: str
    training_rows: int
    source_symbols: list[str]
    target_mode: str | None = None
    label_mode: str | None = None
    horizon: int | None = None
    training_start: str | None = None
    training_end: str | None = None
    benchmark_summary: dict[str, float] | None = None
    simulator_version: str | None = None
    execution_model_version: str | None = None
    train_split: float | None = None
    validation_rows: int | None = None
    sequence_window: int | None = None
    smoothing_alpha: float | None = None
    temporal_model: bool = False
    validation_matrix_summary: dict[str, float] | None = None
    feature_ablation_summary: dict[str, float] | None = None
    model_health_snapshot: dict[str, float | bool] | None = None
    refresh_parent_version: str | None = None
    refresh_reason: str | None = None
    promotion_status: str | None = None


def save_artifact(
    path: str | Path, artifact: RegimeModelArtifact | StrategyValueModelArtifact
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")


def load_regime_model_artifact(path: str | Path) -> RegimeModelArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RegimeModelArtifact(**payload)


def load_strategy_value_model_artifact(path: str | Path) -> StrategyValueModelArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return StrategyValueModelArtifact(**payload)
