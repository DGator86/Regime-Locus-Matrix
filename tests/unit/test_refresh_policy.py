from __future__ import annotations

from rlm.training.refresh_policy import RefreshPolicy, evaluate_refresh_eligibility


def test_refresh_policy_blocks_for_cooldown() -> None:
    policy = RefreshPolicy(min_new_rows=100, cooldown_hours=12.0, require_stale=True)
    out = evaluate_refresh_eligibility(
        trained_at="2099-01-01T00:00:00+00:00",
        new_rows_available=500,
        is_stale=True,
        policy=policy,
    )
    assert out.allowed is False
    assert out.reason == "cooldown_active"
