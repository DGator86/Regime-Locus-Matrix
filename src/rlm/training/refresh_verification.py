from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefreshVerification:
    promote: bool
    reason: str
    selected_realized_improvement: float
    flip_rate_improvement: float


def verify_candidate_promotion(
    *,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
    candidate_health_snapshot: dict[str, float | bool] | None = None,
    require_candidate_not_stale: bool = True,
    min_selected_realized_improvement: float = 0.0,
    min_flip_improvement: float = -0.05,
) -> RefreshVerification:
    sri = float(
        candidate_summary.get("selected_realized_average", 0.0)
        - baseline_summary.get("selected_realized_average", 0.0)
    )
    fri = float(
        baseline_summary.get("regime_flip_rate", 0.0)
        - candidate_summary.get("regime_flip_rate", 0.0)
    )
    if sri < min_selected_realized_improvement:
        return RefreshVerification(False, "selected_realized_regressed", sri, fri)
    if fri < min_flip_improvement:
        return RefreshVerification(False, "flip_rate_worsened_too_much", sri, fri)
    if require_candidate_not_stale and candidate_health_snapshot is not None:
        if bool(candidate_health_snapshot.get("is_stale", False)):
            return RefreshVerification(False, "post_refresh_health_stale", sri, fri)
    return RefreshVerification(True, "candidate_promoted", sri, fri)
