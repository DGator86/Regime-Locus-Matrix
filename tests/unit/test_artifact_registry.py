from __future__ import annotations

from pathlib import Path

from rlm.training.artifact_registry import (
    ArtifactRegistry,
    load_registry,
    rollback_to_previous,
    save_registry,
)


def test_registry_round_trip(tmp_path: Path) -> None:
    reg = ArtifactRegistry(
        active_regime_path="a.json",
        active_value_path="b.json",
        previous_regime_path="c.json",
        previous_value_path="d.json",
        last_refresh_at="2026-04-19T15:00:00Z",
    )
    save_registry(tmp_path, reg)
    loaded = load_registry(tmp_path)
    assert loaded.active_regime_path == "a.json"


def test_rollback_to_previous_swaps_active_and_previous(tmp_path: Path) -> None:
    reg = ArtifactRegistry(
        active_regime_path="active_r.json",
        active_value_path="active_v.json",
        previous_regime_path="prev_r.json",
        previous_value_path="prev_v.json",
    )
    save_registry(tmp_path, reg)
    rolled = rollback_to_previous(tmp_path)
    assert rolled.active_regime_path == "prev_r.json"
    assert rolled.active_value_path == "prev_v.json"
    assert rolled.previous_regime_path == "active_r.json"
    assert rolled.previous_value_path == "active_v.json"
