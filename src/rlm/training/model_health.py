from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class ModelHealthState:
    age_hours: float
    feature_drift_score: float
    regime_drift_score: float
    performance_decay: float
    is_stale: bool


def compute_model_age_hours(trained_at: str) -> float:
    t0 = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - t0).total_seconds() / 3600.0, 0.0)


def compute_feature_drift_score(
    X_train: np.ndarray,
    X_live: np.ndarray,
) -> float:
    """
    Compute normalized mean shift across features.
    """
    x_train = np.asarray(X_train, dtype=float)
    x_live = np.asarray(X_live, dtype=float)
    if x_train.size == 0 or x_live.size == 0:
        return 0.0
    if x_train.ndim != 2 or x_live.ndim != 2:
        raise ValueError("X_train and X_live must be 2D arrays")
    if x_train.shape[1] != x_live.shape[1]:
        raise ValueError("X_train and X_live must have matching feature columns")
    mu_train = x_train.mean(axis=0)
    mu_live = x_live.mean(axis=0)
    std_train = x_train.std(axis=0) + 1e-6
    z_shift = np.abs(mu_live - mu_train) / std_train
    return float(np.mean(z_shift))


def compute_regime_drift_score(
    train_regime_probs: np.ndarray,
    live_regime_probs: np.ndarray,
) -> float:
    """
    KL divergence between average regime distributions.
    """
    train_probs = np.asarray(train_regime_probs, dtype=float)
    live_probs = np.asarray(live_regime_probs, dtype=float)
    if train_probs.size == 0 or live_probs.size == 0:
        return 0.0
    if train_probs.ndim != 2 or live_probs.ndim != 2:
        raise ValueError("train_regime_probs and live_regime_probs must be 2D arrays")
    if train_probs.shape[1] != live_probs.shape[1]:
        raise ValueError("Regime probability arrays must have matching columns")
    eps = 1e-8
    p = train_probs.mean(axis=0)
    q = live_probs.mean(axis=0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_performance_decay(
    realized_returns: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    realized = np.asarray(realized_returns, dtype=float).ravel()
    predicted = np.asarray(predicted_values, dtype=float).ravel()
    if realized.size < 10 or predicted.size < 10:
        return 0.0
    n = min(realized.size, predicted.size)
    realized = realized[:n]
    predicted = predicted[:n]
    corr = float(np.corrcoef(realized, predicted)[0, 1])
    if np.isnan(corr):
        return 0.0
    return float(1.0 - corr)


def evaluate_model_health(
    *,
    trained_at: str,
    X_train: np.ndarray,
    X_live: np.ndarray,
    train_regime_probs: np.ndarray,
    live_regime_probs: np.ndarray,
    realized_returns: np.ndarray,
    predicted_values: np.ndarray,
    max_age_hours: float = 72.0,
    drift_threshold: float = 1.5,
    regime_drift_threshold: float = 0.5,
    decay_threshold: float = 0.6,
) -> ModelHealthState:
    age = compute_model_age_hours(trained_at)
    feature_drift = compute_feature_drift_score(X_train, X_live)
    regime_drift = compute_regime_drift_score(train_regime_probs, live_regime_probs)
    perf_decay = compute_performance_decay(realized_returns, predicted_values)
    is_stale = (
        age > max_age_hours
        or feature_drift > drift_threshold
        or regime_drift > regime_drift_threshold
        or perf_decay > decay_threshold
    )
    return ModelHealthState(
        age_hours=age,
        feature_drift_score=feature_drift,
        regime_drift_score=regime_drift,
        performance_decay=perf_decay,
        is_stale=is_stale,
    )
