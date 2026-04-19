from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlm.training.artifact_registry import promote_candidate
from rlm.training.refresh_verification import verify_candidate_promotion


@dataclass
class RefreshOutcome:
    triggered: bool
    promoted: bool
    reason: str
    candidate_regime_path: str | None = None
    candidate_value_path: str | None = None


def run_refresh_cycle(
    *,
    base_dir: str | Path,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
    candidate_regime_path: Path,
    candidate_value_path: Path,
    promote_on_pass: bool,
    keep_candidate_on_fail: bool,
    min_selected_realized_improvement: float = 0.0,
    min_flip_improvement: float = -0.05,
) -> RefreshOutcome:
    verification = verify_candidate_promotion(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        min_selected_realized_improvement=min_selected_realized_improvement,
        min_flip_improvement=min_flip_improvement,
    )
    if verification.promote and promote_on_pass:
        promote_candidate(
            base_dir=base_dir,
            candidate_regime_path=candidate_regime_path,
            candidate_value_path=candidate_value_path,
        )
        return RefreshOutcome(
            triggered=True,
            promoted=True,
            reason=verification.reason,
            candidate_regime_path=str(candidate_regime_path),
            candidate_value_path=str(candidate_value_path),
        )

    if not keep_candidate_on_fail and not verification.promote:
        candidate_regime_path.unlink(missing_ok=True)
        candidate_value_path.unlink(missing_ok=True)
        parent = candidate_regime_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    return RefreshOutcome(
        triggered=True,
        promoted=False,
        reason=verification.reason,
        candidate_regime_path=str(candidate_regime_path) if keep_candidate_on_fail else None,
        candidate_value_path=str(candidate_value_path) if keep_candidate_on_fail else None,
    )
