from __future__ import annotations

from pathlib import Path

from rlm.training.artifact_registry import ArtifactRegistry, load_registry, save_registry


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
