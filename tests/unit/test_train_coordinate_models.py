from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rlm.training.datasets import build_regime_training_frame, build_strategy_value_training_frame
from rlm.training.train_coordinate_models import (
    load_model_artifacts,
    train_regime_model,
    train_strategy_value_model,
)


def _synthetic_df(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": f"2026-02-01T00:{i:02d}:00Z",
                "symbol": "SPY",
                "close": 100 + i * 0.5,
                "sigma": 0.02,
                "M_D": 5.0 + (i % 4),
                "M_V": 5.0 + (i % 3) * 0.2,
                "M_L": 5.0 + (i % 2) * 0.3,
                "M_G": 5.0 + (i % 5) * 0.1,
                "M_trend_strength": float(i % 3),
                "M_dealer_control": float(i % 2),
                "M_alignment": float((i % 5) - 2),
                "M_delta_neutral": float(i % 4),
                "M_R_trans": float(i % 2),
            }
        )
    return pd.DataFrame(rows)


def test_fitting_on_tiny_synthetic_dataset_succeeds() -> None:
    df = _synthetic_df(30)
    regime_df = build_regime_training_frame(df)
    value_df = build_strategy_value_training_frame(df, horizon=5)

    regime_model = train_regime_model(regime_df)
    value_model = train_strategy_value_model(value_df)

    assert len(regime_model.predict(regime_df.iloc[:3])) == 3
    assert value_model.predict_expected_values(value_df.iloc[:3]).shape == (3, 6)


def test_trained_model_artifacts_are_produced_and_loadable(tmp_path: Path) -> None:
    from rlm.training.train_coordinate_models import save_model_artifacts

    df = _synthetic_df(30)
    regime_model = train_regime_model(build_regime_training_frame(df))
    value_model = train_strategy_value_model(build_strategy_value_training_frame(df, horizon=5))

    save_model_artifacts(
        regime_model,
        value_model,
        regime_training_rows=20,
        value_training_rows=20,
        source_symbols=["SPY"],
        out_dir=tmp_path,
    )

    regime_artifact, value_artifact = load_model_artifacts(tmp_path)
    assert regime_artifact.training_rows == 20
    assert value_artifact.training_rows == 20


def test_loaded_model_predicts_same_as_saved_model(tmp_path: Path) -> None:
    from rlm.features.scoring.regime_model import RegimeModel
    from rlm.roee.strategy_value_model import StrategyValueModel
    from rlm.training.artifacts import load_regime_model_artifact, load_strategy_value_model_artifact
    from rlm.training.train_coordinate_models import save_model_artifacts

    df = _synthetic_df(40)
    regime_df = build_regime_training_frame(df)
    value_df = build_strategy_value_training_frame(df, horizon=5)
    regime_model = train_regime_model(regime_df)
    value_model = train_strategy_value_model(value_df)

    regime_path, value_path = save_model_artifacts(
        regime_model,
        value_model,
        regime_training_rows=len(regime_df),
        value_training_rows=len(value_df),
        source_symbols=["SPY"],
        out_dir=tmp_path,
    )

    loaded_regime = RegimeModel.from_artifact(load_regime_model_artifact(regime_path))
    loaded_value = StrategyValueModel.from_artifact(load_strategy_value_model_artifact(value_path))

    x_reg = regime_df.iloc[:5]
    x_val = value_df.iloc[:5]
    assert regime_model.predict(x_reg) == loaded_regime.predict(x_reg)
    assert np.allclose(
        value_model.predict_expected_values(x_val),
        loaded_value.predict_expected_values(x_val),
    )
