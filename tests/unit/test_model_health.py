from __future__ import annotations

import numpy as np

from rlm.training.model_health import (
    compute_feature_drift_score,
    evaluate_model_health,
)
from rlm.training.retrain_trigger import should_trigger_retrain


def test_feature_drift_detects_shift() -> None:
    rng = np.random.default_rng(seed=7)
    x1 = rng.normal(0, 1, (100, 5))
    x2 = rng.normal(2, 1, (100, 5))
    score = compute_feature_drift_score(x1, x2)
    assert score > 1.0


def test_health_flags_stale_model() -> None:
    health = evaluate_model_health(
        trained_at="2000-01-01T00:00:00Z",
        X_train=np.zeros((10, 3)),
        X_live=np.ones((10, 3)),
        train_regime_probs=np.ones((10, 2)),
        live_regime_probs=np.ones((10, 2)) * 0.5,
        realized_returns=np.ones(20),
        predicted_values=np.zeros(20),
    )
    assert health.is_stale
    assert should_trigger_retrain(health)
