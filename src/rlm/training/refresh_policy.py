from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class RefreshPolicy:
    min_new_rows: int = 250
    cooldown_hours: float = 12.0
    require_stale: bool = True


@dataclass(frozen=True)
class RefreshEligibility:
    allowed: bool
    reason: str
    new_rows_available: int
    hours_since_last_train: float


def hours_since(trained_at: str) -> float:
    t0 = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return max((now - t0).total_seconds() / 3600.0, 0.0)


def evaluate_refresh_eligibility(
    *,
    trained_at: str,
    new_rows_available: int,
    is_stale: bool,
    policy: RefreshPolicy,
) -> RefreshEligibility:
    age = hours_since(trained_at)
    if policy.require_stale and not is_stale:
        return RefreshEligibility(False, "model_not_stale", new_rows_available, age)
    if new_rows_available < policy.min_new_rows:
        return RefreshEligibility(False, "insufficient_new_rows", new_rows_available, age)
    if age < policy.cooldown_hours:
        return RefreshEligibility(False, "cooldown_active", new_rows_available, age)
    return RefreshEligibility(True, "eligible", new_rows_available, age)
