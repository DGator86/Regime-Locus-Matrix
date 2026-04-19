from __future__ import annotations

from rlm.training.refresh_verification import verify_candidate_promotion


def test_verify_candidate_rejects_regression() -> None:
    out = verify_candidate_promotion(
        baseline_summary={"selected_realized_average": 0.10, "regime_flip_rate": 0.20},
        candidate_summary={"selected_realized_average": 0.08, "regime_flip_rate": 0.18},
    )
    assert out.promote is False


def test_verify_candidate_rejects_stale_post_refresh_health() -> None:
    out = verify_candidate_promotion(
        baseline_summary={"selected_realized_average": 0.10, "regime_flip_rate": 0.20},
        candidate_summary={"selected_realized_average": 0.12, "regime_flip_rate": 0.18},
        candidate_health_snapshot={"is_stale": True},
    )
    assert out.promote is False
    assert out.reason == "post_refresh_health_stale"
