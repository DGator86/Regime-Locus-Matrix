"""Transition probability calibration I/O and transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlm.regimes.transition_calibration import (
    TransitionProbabilityCalibration,
    apply_top1_calibration_inplace,
    load_transition_calibration,
    save_calibration,
)


def test_calibration_save_load_roundtrip(tmp_path: Path) -> None:
    payload = {
        "kind": "isotonic_top1_next_regime",
        "regime_family": "hmm",
        "x_thresholds": [0.0, 0.5, 1.0],
        "y_thresholds": [0.1, 0.4, 0.9],
        "n_samples": 100,
    }
    path = tmp_path / "cal.json"
    save_calibration(payload, path)
    cal = load_transition_calibration(path=path)
    assert cal is not None
    assert cal.regime_family == "hmm"
    np.testing.assert_allclose(cal.transform(np.array([0.25, 0.75])), [0.25, 0.65], atol=1e-9)


def test_apply_top1_inplace_skips_when_column_missing(tmp_path: Path) -> None:
    import pandas as pd

    cal = TransitionProbabilityCalibration(
        kind="isotonic_top1_next_regime",
        x_thresholds=np.array([0.0, 1.0]),
        y_thresholds=np.array([0.0, 1.0]),
        regime_family="hmm",
        meta={},
    )
    df = pd.DataFrame({"a": [1.0]})
    apply_top1_calibration_inplace(df, "missing", "out", cal)
    assert "out" not in df.columns


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)
def test_fit_isotonic_produces_payload() -> None:
    from rlm.regimes.transition_calibration import fit_isotonic_top1_next_regime

    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.2, 0.95, size=n)
    pred = rng.integers(0, 3, size=n)
    actual = pred.copy()
    flip = rng.random(n) < 0.35
    actual[flip] = (actual[flip] + 1) % 3
    out = fit_isotonic_top1_next_regime(p, pred, actual, regime_family="hmm")
    assert out is not None
    assert len(out["x_thresholds"]) >= 2
    assert json.dumps(out)
