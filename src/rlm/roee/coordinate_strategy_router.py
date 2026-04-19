from __future__ import annotations

from pathlib import Path
from typing import Mapping

from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.training.artifacts import load_strategy_value_model_artifact


def load_trained_value_model_or_bootstrap(
    path: str | Path = "artifacts/models/strategy_value_model.json",
) -> StrategyValueModel:
    artifact_path = Path(path)
    if artifact_path.exists():
        return StrategyValueModel.from_artifact(load_strategy_value_model_artifact(artifact_path))
    return StrategyValueModel.with_bootstrap_coefficients()


_DEFAULT_VALUE_MODEL = load_trained_value_model_or_bootstrap()


def select_strategy_from_coordinates(row: Mapping[str, float]) -> str:
    scores = _DEFAULT_VALUE_MODEL.score_row(row)
    return scores.best_strategy
