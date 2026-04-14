from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline
from rlm.optimization.config import NightlyHyperparams
from rlm.roee.engine import ROEEConfig


class OptimizationBase:
    """Shared utilities for nightly hyperparameter optimization."""

    @staticmethod
    def load_bars(
        symbol: str, lookback_bars: int = 252 * 2, root: Path | None = None
    ) -> pd.DataFrame:
        """Load daily bars from data/raw/bars_<SYMBOL>.csv."""
        repo_root = root or Path(__file__).resolve().parents[3]
        bars_path = repo_root / "data" / "raw" / f"bars_{symbol.upper()}.csv"
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing bars file: {bars_path}")

        bars = pd.read_csv(bars_path)
        if "timestamp" in bars.columns:
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")

        return bars.tail(lookback_bars).copy()

    @staticmethod
    def objective(trial: optuna.Trial, symbols: list[str], regime_model: str) -> float:
        nightly = NightlyHyperparams(
            mtf_ltf_weight=trial.suggest_float("mtf_ltf_weight", 0.35, 0.65),
            mtf_regimes=trial.suggest_categorical("mtf_regimes", [True, False]),
            hmm_confidence_threshold=trial.suggest_float("hmm_confidence_threshold", 0.55, 0.75),
            high_vol_kelly_multiplier=trial.suggest_float("high_vol_kelly_multiplier", 0.45, 0.75),
            transition_kelly_multiplier=trial.suggest_float(
                "transition_kelly_multiplier", 0.70, 0.95
            ),
            calm_trend_kelly_multiplier=trial.suggest_float(
                "calm_trend_kelly_multiplier", 1.05, 1.35
            ),
            move_window=trial.suggest_int("move_window", 85, 115),
            vol_window=trial.suggest_int("vol_window", 85, 115),
            direction_neutral_threshold=trial.suggest_float(
                "direction_neutral_threshold", 0.26, 0.34
            ),
        )

        cfg = FullRLMConfig(
            regime_model=regime_model,
            mtf=True,
            roee_config=ROEEConfig(use_dynamic_sizing=True),
            nightly_hyperparams=nightly.__dict__,
        )

        scores: list[float] = []
        for sym in symbols:
            try:
                bars = OptimizationBase.load_bars(sym, lookback_bars=252 * 2)
            except FileNotFoundError:
                continue
            result = FullRLMPipeline(cfg).run(bars)
            if result.policy_df.empty:
                continue
            tail = result.policy_df.tail(126)
            enter_frac = (
                float((tail["roee_action"] == "enter").mean()) if "roee_action" in tail else 0.0
            )
            confidence = float(tail.get("regime_confidence", pd.Series([0.0])).fillna(0).mean())
            score = confidence * (1.0 + enter_frac)
            scores.append(score)

        return float(pd.Series(scores).mean()) if scores else -999.0
