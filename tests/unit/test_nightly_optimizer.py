from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from rlm.optimization import nightly


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
