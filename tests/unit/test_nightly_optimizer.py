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


def test_nightly_optimizer_scrubs_stale_mtf_regimes_when_preserving_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = {"move_window": 91, "mtf_regimes": True, "transaction_cost_frac": 0.0015}
def test_nightly_optimizer_scrubs_unsafe_existing_overlay_when_all_trials_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = {"move_window": 91, "mtf_regimes": True}
    nightly_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    nightly_path.parent.mkdir(parents=True)
    nightly_path.write_text(json.dumps(existing), encoding="utf-8")

    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", nightly_path)

    def prune_objective(*_args: object, **_kwargs: object) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", prune_objective)

    out = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert out == {"move_window": 91, "transaction_cost_frac": 0.0015}
    assert json.loads(nightly_path.read_text(encoding="utf-8")) == out
    assert out == {"move_window": 91}
    assert json.loads(nightly_path.read_text(encoding="utf-8")) == {"move_window": 91}


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



def test_nightly_optimizer_ignores_malformed_existing_overlay_when_all_trials_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nightly_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    nightly_path.parent.mkdir(parents=True)
    nightly_path.write_text("{", encoding="utf-8")

    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_regime.json")
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", nightly_path)

    def prune_objective(*_args: object, **_kwargs: object) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", prune_objective)

    out = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=2)

    assert out == {}
    assert nightly_path.read_text(encoding="utf-8") == "{"


def test_nightly_optimizer_does_not_write_overlay_without_valid_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", out_path)
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_live_regime_model.json")
    monkeypatch.setattr(
        nightly.OptimizationBase,
        "objective",
        staticmethod(lambda trial, symbols, regime_model: nightly.NO_VALID_SCORE),
    )

    with pytest.raises(RuntimeError, match="no valid backtest scores"):
        nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=1)

    assert not out_path.exists()


def test_nightly_optimizer_writes_overlay_for_valid_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", out_path)
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_live_regime_model.json")

    def _objective(trial, symbols, regime_model) -> float:
        trial.suggest_int("move_window", 90, 90)
        return nightly.NO_VALID_SCORE + 1.0

    monkeypatch.setattr(nightly.OptimizationBase, "objective", staticmethod(_objective))

    best = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=1)

    assert best == {"move_window": 90}
    assert '"move_window": 90' in out_path.read_text(encoding="utf-8")


def test_nightly_optimizer_returns_empty_when_all_trials_are_pruned(
def test_nightly_optimizer_excludes_mtf_regimes_from_written_live_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "data" / "processed" / "live_nightly_hyperparams.json"
    monkeypatch.setattr(nightly, "NIGHTLY_PATH", out_path)
    monkeypatch.setattr(nightly, "REGIME_PATH", tmp_path / "missing_live_regime_model.json")

    def _pruned_objective(trial, symbols, regime_model) -> float:
        raise optuna.TrialPruned()

    monkeypatch.setattr(nightly.OptimizationBase, "objective", staticmethod(_pruned_objective))

    assert nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=1) == {}
    assert not out_path.exists()
    def _objective(trial, symbols, regime_model) -> float:
        trial.suggest_int("move_window", 90, 90)
        trial.suggest_categorical("mtf_regimes", [True])
        return nightly.NO_VALID_SCORE + 1.0

    monkeypatch.setattr(nightly.OptimizationBase, "objective", staticmethod(_objective))

    best = nightly.NightlyMTFOptimizer.run(symbols=["SPY"], trials=1)

    assert best == {"move_window": 90}
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"move_window": 90}
