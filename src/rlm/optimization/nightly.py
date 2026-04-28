from __future__ import annotations

import json
from pathlib import Path

import optuna

from .base import OptimizationBase

ROOT = Path(__file__).resolve().parents[3]
REGIME_PATH = ROOT / "data/processed/live_regime_model.json"
NIGHTLY_PATH = ROOT / "data/processed/live_nightly_hyperparams.json"


class NightlyMTFOptimizer:
    """Structural nightly optimizer that never mutates the weekly regime model."""

    @staticmethod
    def run(symbols: list[str] | None = None, trials: int = 40) -> dict:
        symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

        regime_model = "hmm"
        if REGIME_PATH.exists():
            model = json.loads(REGIME_PATH.read_text(encoding="utf-8"))
            if isinstance(model, dict):
                regime_model = str(model.get("model", model.get("regime_model", regime_model)))

        study = optuna.create_study(
            direction="maximize",
            study_name="nightly_mtf",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda trial: OptimizationBase.objective(trial, symbols, regime_model),
            n_trials=trials,
            timeout=3600,
        )

        best = study.best_params
        NIGHTLY_PATH.parent.mkdir(parents=True, exist_ok=True)
        NIGHTLY_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
        return best
