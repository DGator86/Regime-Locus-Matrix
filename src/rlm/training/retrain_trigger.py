from __future__ import annotations

from rlm.training.model_health import ModelHealthState


def should_trigger_retrain(health: ModelHealthState) -> bool:
    return bool(health.is_stale)
