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
LIVE_OVERLAY_KEYS = {
    "mtf_ltf_weight",
    "hmm_confidence_threshold",
    "high_vol_kelly_multiplier",
    "transition_kelly_multiplier",
    "calm_trend_kelly_multiplier",
    "move_window",
    "vol_window",
    "direction_neutral_threshold",
    "transaction_cost_frac",
}


def _live_overlay_params(params: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in params.items() if key in LIVE_OVERLAY_KEYS}


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
                try:
                    existing = json.loads(NIGHTLY_PATH.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {}
                if not isinstance(existing, dict):
                    return {}
                safe_existing = _live_overlay_params(existing)
                if safe_existing != existing:
                    tmp_path = NIGHTLY_PATH.with_suffix(f"{NIGHTLY_PATH.suffix}.tmp")
                    tmp_path.write_text(json.dumps(safe_existing, indent=2), encoding="utf-8")
                    tmp_path.replace(NIGHTLY_PATH)
                return safe_existing
            return {}

        if float(study.best_value) <= NO_VALID_SCORE:
            raise RuntimeError(
                "Nightly optimization produced no valid backtest scores; "
                "leaving live_nightly_hyperparams.json unchanged."
            )

        best = _live_overlay_params(study.best_params)

        NIGHTLY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = NIGHTLY_PATH.with_suffix(f"{NIGHTLY_PATH.suffix}.tmp")
        tmp_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        tmp_path.replace(NIGHTLY_PATH)
        return best
