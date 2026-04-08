"""Markov-switching overlay for RLM using statsmodels."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class MarkovSwitchingConfig(BaseModel):
    n_states: int = Field(
        3,
        ge=2,
        le=6,
        description="Number of Markov regimes",
    )
    switching_variance: bool = True
    trend: str = "c"
    model_path: Path | None = None


class RLMMarkovSwitching:
    def __init__(self, config: MarkovSwitchingConfig = MarkovSwitchingConfig()):
        self.config = config
        self.fit_result = None
        self.state_labels: list[str] | None = None

    @staticmethod
    def prepare_endog(df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("Missing required 'close' column for Markov-switching observations.")
        close = pd.to_numeric(df["close"], errors="coerce")
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return returns.astype(float)

    def fit(self, df_train: pd.DataFrame, verbose: bool = True) -> "RLMMarkovSwitching":
        endog = self.prepare_endog(df_train)
        model = MarkovRegression(
            endog=endog,
            k_regimes=self.config.n_states,
            trend=self.config.trend,
            switching_variance=self.config.switching_variance,
        )
        self.fit_result = model.fit(disp=False)
        if verbose:
            print(
                f"Markov-switching fitted with {self.config.n_states} states, "
                f"log-likelihood: {float(self.fit_result.llf):.2f}"
            )
        self._auto_label_states(df_train, self.fit_result.filtered_marginal_probabilities)
        return self

    def _auto_label_states(self, df: pd.DataFrame, probs: pd.DataFrame) -> None:
        state_idx = probs.to_numpy().argmax(axis=1)
        labels: list[str] = []
        for s in range(self.config.n_states):
            mask = state_idx == s
            if mask.sum() == 0:
                labels.append(f"state_{s}")
                continue
            if "regime_key" in df.columns and not df.loc[mask, "regime_key"].dropna().empty:
                labels.append(f"{str(df.loc[mask, 'regime_key'].mode().iloc[0])}_like")
            else:
                labels.append(f"state_{s}")
        self.state_labels = labels

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fit_result is None:
            raise RuntimeError("Markov-switching model is not fitted")
        endog = self.prepare_endog(df)
        model = MarkovRegression(
            endog=endog,
            k_regimes=self.config.n_states,
            trend=self.config.trend,
            switching_variance=self.config.switching_variance,
        )
        result = model.filter(self.fit_result.params)
        probs = result.filtered_marginal_probabilities.copy()
        probs.index = df.index
        return probs

    def most_likely_state_filtered(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.filter(df)
        return probs.to_numpy().argmax(axis=1).astype(int)

    def annotate(self, df: pd.DataFrame, prefix: str = "markov") -> pd.DataFrame:
        probs = self.filter(df)
        out = df.copy()
        out[f"{prefix}_probs"] = probs.to_numpy().tolist()
        out[f"{prefix}_state"] = probs.to_numpy().argmax(axis=1).astype(int)
        out[f"{prefix}_confidence"] = probs.max(axis=1).astype(float).to_numpy()
        if self.state_labels:
            out[f"{prefix}_state_label"] = [
                self.state_labels[int(s)] for s in out[f"{prefix}_state"]
            ]
        return out

    def save(self, path: Path | None = None) -> None:
        path = path or self.config.model_path or Path("models/rlm_markov_switching.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RLMMarkovSwitching":
        return joblib.load(path)
