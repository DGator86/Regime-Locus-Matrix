from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.training.artifact_registry import load_registry
from rlm.training.artifacts import load_strategy_value_model_artifact

logger = logging.getLogger(__name__)


def _has_valid_metadata(artifact) -> bool:
    return (
        isinstance(getattr(artifact, "label_mode", None), str)
        and isinstance(getattr(artifact, "target_mode", None), str)
        and isinstance(getattr(artifact, "horizon", None), int)
        and artifact.horizon > 0
    )


def _resolve_active_value_artifact(base_dir: str | Path = "artifacts/models") -> Path | None:
    reg = load_registry(base_dir)
    if reg.active_value_path:
        path = Path(reg.active_value_path)
        if path.exists():
            return path
    fallback = Path(base_dir) / "strategy_value_model.json"
    return fallback if fallback.exists() else None


def load_trained_value_model_or_bootstrap(
    path: str | Path | None = None,
) -> StrategyValueModel:
    artifact_path = Path(path) if path is not None else _resolve_active_value_artifact()
    if artifact_path is not None and artifact_path.exists():
        artifact = load_strategy_value_model_artifact(artifact_path)
        if _has_valid_metadata(artifact):
            return StrategyValueModel.from_artifact(artifact)
        logger.warning("Strategy artifact metadata missing/malformed at %s; using bootstrap", artifact_path)
    return StrategyValueModel.with_bootstrap_coefficients()


_DEFAULT_VALUE_MODEL = load_trained_value_model_or_bootstrap()


def select_strategy_from_coordinates(row: Mapping[str, float]) -> str:
    scores = _DEFAULT_VALUE_MODEL.score_row(row)
    return scores.best_strategy
