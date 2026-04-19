from __future__ import annotations

from pathlib import Path

from rlm.training.refresh_controller import run_refresh_cycle


def test_refresh_cycle_returns_rejected_when_candidate_fails_verification(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidates" / "v1"
    candidate_dir.mkdir(parents=True)
    regime = candidate_dir / "regime_model.json"
    value = candidate_dir / "strategy_value_model.json"
    regime.write_text("{}", encoding="utf-8")
    value.write_text("{}", encoding="utf-8")

    outcome = run_refresh_cycle(
        base_dir=tmp_path,
        baseline_summary={"selected_realized_average": 0.12, "regime_flip_rate": 0.2},
        candidate_summary={"selected_realized_average": 0.08, "regime_flip_rate": 0.19},
        candidate_regime_path=regime,
        candidate_value_path=value,
        promote_on_pass=True,
        keep_candidate_on_fail=False,
    )

    assert outcome.triggered is True
    assert outcome.promoted is False
    assert outcome.reason == "selected_realized_regressed"
    assert not regime.exists()
    assert not value.exists()
