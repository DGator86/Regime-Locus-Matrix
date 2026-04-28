from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from rlm.optimization import nightly


def test_nightly_optimizer_preserves_existing_overlay_when_all_trials_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = {"move_window": 91, "transaction_cost_frac": 0.0015}
    nightly_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    nightly_path.parent.mkdir(parents=True)
    nightly_path.write_text(json.dumps(existing), encoding="utf-8")

    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", nightly_path)

    def prune_objective(*_args: object, **_kwargs: object) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", prune_objective)

    out = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert out == existing
    assert json.loads(nightly_path.read_text(encoding="utf-8")) == existing


def test_nightly_optimizer_returns_empty_overlay_when_all_trials_pruned_without_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nightly_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", nightly_path)

    def prune_objective(*_args: object, **_kwargs: object) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", prune_objective)

    out = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert out == {}
    assert not nightly_path.exists()
