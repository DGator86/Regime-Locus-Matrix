from __future__ import annotations

from pathlib import Path

from rlm.features.scoring.coordinate_regime import _resolve_active_regime_artifact
from rlm.roee.coordinate_strategy_router import _resolve_active_value_artifact
from rlm.training.artifact_registry import ArtifactRegistry, save_registry


def test_runtime_resolves_active_artifacts_from_registry(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates" / "20260419T010101Z"
    candidate_dir.mkdir(parents=True)
    regime = candidate_dir / "regime_model.json"
    value = candidate_dir / "strategy_value_model.json"
    regime.write_text("{}", encoding="utf-8")
    value.write_text("{}", encoding="utf-8")

    save_registry(
        tmp_path,
        ArtifactRegistry(active_regime_path=str(regime), active_value_path=str(value)),
    )

    assert _resolve_active_regime_artifact(tmp_path) == regime
    assert _resolve_active_value_artifact(tmp_path) == value
