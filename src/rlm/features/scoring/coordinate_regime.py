from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from rlm.features.scoring.regime_model import REGIME_LABELS
from rlm.features.scoring.regime_model import RegimeModel
from rlm.training.artifacts import load_regime_model_artifact

logger = logging.getLogger(__name__)

RegimeLabel = Literal[
    "trend_up_stable",
    "trend_down_stable",
    "range_compression",
    "mean_reversion",
    "breakout_expansion",
    "transition",
    "chaos",
    "no_trade",
]


def _has_valid_metadata(artifact) -> bool:
    return (
        isinstance(getattr(artifact, "label_mode", None), str)
        and isinstance(getattr(artifact, "target_mode", None), str)
        and isinstance(getattr(artifact, "horizon", None), int)
        and artifact.horizon > 0
    )


def load_trained_regime_model_or_bootstrap(
    path: str | Path = "artifacts/models/regime_model.json",
) -> RegimeModel:
    artifact_path = Path(path)
    if artifact_path.exists():
        artifact = load_regime_model_artifact(artifact_path)
        if _has_valid_metadata(artifact):
            return RegimeModel.from_artifact(artifact)
        logger.warning("Regime artifact metadata missing/malformed at %s; using bootstrap", artifact_path)
    return RegimeModel.with_bootstrap_coefficients()


_DEFAULT_REGIME_MODEL = load_trained_regime_model_or_bootstrap()


def score_regime_probabilities_from_coordinates(
    row: Mapping[str, float],
) -> dict[RegimeLabel, float]:
    labels = tuple(REGIME_LABELS)
    state = np.array(
        [
            [
                _v(row, "M_D"),
                _v(row, "M_V"),
                _v(row, "M_L"),
                _v(row, "M_G"),
                _v(row, "M_trend_strength"),
                _v(row, "M_dealer_control"),
                _v(row, "M_alignment"),
                _v(row, "M_delta_neutral"),
                _v(row, "M_R_trans"),
            ]
        ],
        dtype=float,
    )
    probs = _DEFAULT_REGIME_MODEL.predict_proba(state)[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


def classify_regime_from_coordinates(row: Mapping[str, float]) -> RegimeLabel:
    probs = score_regime_probabilities_from_coordinates(row)
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _v(row: Mapping[str, float], key: str) -> float:
    raw = row.get(key, 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value
