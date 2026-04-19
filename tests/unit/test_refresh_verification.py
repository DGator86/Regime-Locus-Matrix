from __future__ import annotations

from rlm.training.refresh_verification import verify_candidate_promotion


def test_verify_candidate_rejects_regression() -> None:
    out = verify_candidate_promotion(
        baseline_summary={"selected_realized_average": 0.10, "regime_flip_rate": 0.20},
        candidate_summary={"selected_realized_average": 0.08, "regime_flip_rate": 0.18},
    )
    assert out.promote is False
