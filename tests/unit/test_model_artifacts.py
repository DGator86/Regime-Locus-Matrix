from __future__ import annotations

from pathlib import Path

from rlm.features.scoring.regime_model import RegimeModel
from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.training.artifacts import load_regime_model_artifact, load_strategy_value_model_artifact
from rlm.training.train_coordinate_models import (
    load_trained_regime_model_or_bootstrap,
    load_trained_value_model_or_bootstrap,
    save_model_artifacts,
)


def test_model_artifact_round_trip(tmp_path: Path) -> None:
    regime_model = RegimeModel.with_bootstrap_coefficients()
    value_model = StrategyValueModel.with_bootstrap_coefficients()

    regime_path, value_path = save_model_artifacts(
        regime_model,
        value_model,
        regime_training_rows=10,
        value_training_rows=12,
        source_symbols=["SPY"],
        out_dir=tmp_path,
    )

    regime_loaded = RegimeModel.from_artifact(load_regime_model_artifact(regime_path))
    value_loaded = StrategyValueModel.from_artifact(load_strategy_value_model_artifact(value_path))

    row = {
        "M_D": 8.0,
        "M_V": 4.0,
        "M_L": 7.0,
        "M_G": 8.0,
        "M_trend_strength": 3.0,
        "M_dealer_control": 3.0,
        "M_alignment": 9.0,
        "M_delta_neutral": 5.0,
        "M_R_trans": 1.0,
    }

    assert regime_model.predict([list(row.values())]) == regime_loaded.predict([list(row.values())])
    assert value_model.score_row(row).best_strategy == value_loaded.score_row(row).best_strategy


def test_runtime_loader_falls_back_to_bootstrap_when_missing(tmp_path: Path) -> None:
    regime_model = load_trained_regime_model_or_bootstrap(tmp_path / "missing_regime.json")
    value_model = load_trained_value_model_or_bootstrap(tmp_path / "missing_value.json")

    assert isinstance(regime_model, RegimeModel)
    assert isinstance(value_model, StrategyValueModel)


def test_model_artifact_persists_health_snapshot(tmp_path: Path) -> None:
    regime_model = RegimeModel.with_bootstrap_coefficients()
    value_model = StrategyValueModel.with_bootstrap_coefficients()
    health_snapshot = {
        "age_hours": 1.0,
        "feature_drift_score": 0.2,
        "regime_drift_score": 0.1,
        "performance_decay": 0.3,
        "is_stale": False,
    }

    regime_path, value_path = save_model_artifacts(
        regime_model,
        value_model,
        regime_training_rows=10,
        value_training_rows=12,
        source_symbols=["SPY"],
        out_dir=tmp_path,
        model_health_snapshot=health_snapshot,
    )

    regime_artifact = load_regime_model_artifact(regime_path)
    value_artifact = load_strategy_value_model_artifact(value_path)
    assert regime_artifact.model_health_snapshot == health_snapshot
    assert value_artifact.model_health_snapshot == health_snapshot
