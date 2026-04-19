from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from rlm.features.scoring.regime_model import REGIME_LABELS, RegimeModel
from rlm.training.artifact_registry import load_registry
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


def _resolve_active_regime_artifact(base_dir: str | Path = "artifacts/models") -> Path | None:
    reg = load_registry(base_dir)
    if reg.active_regime_path:
        path = Path(reg.active_regime_path)
        if path.exists():
            return path
    fallback = Path(base_dir) / "regime_model.json"
    return fallback if fallback.exists() else None


def load_trained_regime_model_or_bootstrap(
    path: str | Path | None = None,
) -> RegimeModel:
    artifact_path = Path(path) if path is not None else _resolve_active_regime_artifact()
    if artifact_path is not None and artifact_path.exists():
        artifact = load_regime_model_artifact(artifact_path)
        if _has_valid_metadata(artifact):
            return RegimeModel.from_artifact(artifact)
        logger.warning(
            "Regime artifact metadata missing/malformed at %s; using bootstrap", artifact_path
        )
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
