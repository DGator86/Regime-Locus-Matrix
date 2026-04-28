"""Tests for nightly hyperparameter optimization orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import optuna

from rlm.optimization import nightly
from rlm.optimization.config import NightlyHyperparams


def test_nightly_optimizer_uses_defaults_when_all_trials_pruned(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", output_path)

    def always_pruned(trial: optuna.Trial, symbols: list[str], regime_model: str) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", always_pruned)

    best = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert best == NightlyHyperparams().__dict__
    assert json.loads(output_path.read_text(encoding="utf-8")) == best


def test_nightly_optimizer_preserves_existing_overlay_when_all_trials_pruned(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    output_path.parent.mkdir(parents=True)
    existing = {"move_window": 93, "transaction_cost_frac": 0.002}
    output_path.write_text(json.dumps(existing), encoding="utf-8")
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", output_path)

    def always_pruned(trial: optuna.Trial, symbols: list[str], regime_model: str) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", always_pruned)

    best = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert best == existing
    assert json.loads(output_path.read_text(encoding="utf-8")) == existing
