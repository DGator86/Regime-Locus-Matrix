from __future__ import annotations

import json
from pathlib import Path

import optuna
from optuna.trial import TrialState

from .base import OptimizationBase

ROOT = Path(__file__).resolve().parents[3]
REGIME_PATH = ROOT / "data/processed/live_regime_model.json"
NIGHTLY_PATH = ROOT / "data/processed/live_nightly_hyperparams.json"
NO_VALID_SCORE = -999.0


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

        completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not completed:
            if NIGHTLY_PATH.exists():
                existing = json.loads(NIGHTLY_PATH.read_text(encoding="utf-8"))
                return existing if isinstance(existing, dict) else {}
            return {}

        if float(study.best_value) <= NO_VALID_SCORE:
            raise RuntimeError(
                "Nightly optimization produced no valid backtest scores; "
                "leaving live_nightly_hyperparams.json unchanged."
            )

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(
                "[NightlyMTFOptimizer] All trials were pruned — no valid OOS scores. "
                "Check that bars files exist in data/raw/ and the pipeline runs correctly. "
                "Skipping hyperparams write.",
                flush=True,
            )
            return {}
        best = study.best_params
        NIGHTLY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = NIGHTLY_PATH.with_suffix(f"{NIGHTLY_PATH.suffix}.tmp")
        tmp_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        tmp_path.replace(NIGHTLY_PATH)
        return best
