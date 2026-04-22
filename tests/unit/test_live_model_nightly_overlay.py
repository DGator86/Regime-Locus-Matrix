"""Nightly hyperparameter JSON overlay into live regime config."""

from __future__ import annotations

import json
from pathlib import Path

from rlm.forecasting.live_model import LiveRegimeModelConfig, apply_nightly_hyperparam_overlay


def test_overlay_applies_known_keys(tmp_path: Path) -> None:
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "live_nightly_hyperparams.json").write_text(
        json.dumps(
            {
                "move_window": 90,
                "vol_window": 95,
                "direction_neutral_threshold": 0.28,
                "hmm_confidence_threshold": 0.62,
            }
        ),
        encoding="utf-8",
    )
    base = LiveRegimeModelConfig()
    out = apply_nightly_hyperparam_overlay(base, tmp_path)
    assert out.forecast.move_window == 90
    assert out.forecast.vol_window == 95
    assert out.forecast.direction_neutral_threshold == 0.28
    assert out.roee.confidence_threshold == 0.62


def test_overlay_missing_file_returns_same(tmp_path: Path) -> None:
    base = LiveRegimeModelConfig()
    out = apply_nightly_hyperparam_overlay(base, tmp_path)
    assert out == base
