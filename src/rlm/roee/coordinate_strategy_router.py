from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.training.artifacts import load_strategy_value_model_artifact

logger = logging.getLogger(__name__)


def _has_valid_metadata(artifact) -> bool:
    return (
        isinstance(getattr(artifact, "label_mode", None), str)
        and isinstance(getattr(artifact, "target_mode", None), str)
        and isinstance(getattr(artifact, "horizon", None), int)
        and artifact.horizon > 0
    )


def load_trained_value_model_or_bootstrap(
    path: str | Path = "artifacts/models/strategy_value_model.json",
) -> StrategyValueModel:
    artifact_path = Path(path)
    if artifact_path.exists():
        artifact = load_strategy_value_model_artifact(artifact_path)
        if _has_valid_metadata(artifact):
            return StrategyValueModel.from_artifact(artifact)
        logger.warning("Strategy artifact metadata missing/malformed at %s; using bootstrap", artifact_path)
    return StrategyValueModel.with_bootstrap_coefficients()


_DEFAULT_VALUE_MODEL = load_trained_value_model_or_bootstrap()


def select_strategy_from_coordinates(row: Mapping[str, float]) -> str:
    scores = _DEFAULT_VALUE_MODEL.score_row(row)
    return scores.best_strategy
