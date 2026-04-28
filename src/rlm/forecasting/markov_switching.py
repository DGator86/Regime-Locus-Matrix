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
        le=15,
        description="Number of Markov regimes",
    )
    switching_variance: bool = True
    trend: str = "c"
    model_path: Path | None = None
    transition_pseudocount: float = Field(
        0.1,
        ge=0.0,
        description="Symmetric smoothing on the Markov transition matrix rows (calibrated P(i→j)).",
    )
    use_intraday_vp_features: bool = False
    use_wyckoff_features: bool = False
    use_confluence_features: bool = False


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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Build optional exogenous features for regime conditioning."""

        features = pd.DataFrame(index=df.index)

        if "auction_state" in df.columns:
            auction = (
                df["auction_state"]
                .astype("string")
                .fillna("balance")
                .str.lower()
                .replace({"imbalance-up": "imbalance_up", "imbalance-down": "imbalance_down"})
            )
            dummies = pd.get_dummies(auction, prefix="auction_state")
            for name in (
                "auction_state_balance",
                "auction_state_imbalance_up",
                "auction_state_imbalance_down",
            ):
                features[name] = dummies.get(name, 0.0).astype(float)

        if "effort_result_divergence" in df.columns:
            features["effort_result_divergence"] = pd.to_numeric(
                df["effort_result_divergence"], errors="coerce"
            ).fillna(0.0)

        if self.config.use_intraday_vp_features:
            if {"close", "vp_poc"}.issubset(df.columns):
                close = pd.to_numeric(df["close"], errors="coerce")
                vp_poc = pd.to_numeric(df["vp_poc"], errors="coerce")
                denom = close.replace(0.0, np.nan)
                features["vp_poc_distance"] = ((close - vp_poc) / denom).fillna(0.0)
            if {"close", "vp_va_low", "vp_va_high"}.issubset(df.columns):
                close = pd.to_numeric(df["close"], errors="coerce")
                va_low = pd.to_numeric(df["vp_va_low"], errors="coerce")
                va_high = pd.to_numeric(df["vp_va_high"], errors="coerce")
                va_range = (va_high - va_low).replace(0.0, np.nan)
                features["vp_va_position"] = ((close - va_low) / va_range).fillna(0.0)

        if self.config.use_wyckoff_features and "cumulative_wyckoff_score" in df.columns:
            features["cumulative_wyckoff_score"] = pd.to_numeric(
                df["cumulative_wyckoff_score"], errors="coerce"
            ).fillna(0.0)

        if self.config.use_confluence_features:
            if "vp_hybrid_strength_max" in df.columns:
                features["vp_hybrid_strength_max"] = pd.to_numeric(
                    df["vp_hybrid_strength_max"], errors="coerce"
                ).fillna(0.0)
            if "vp_gex_confluence_poc" in df.columns:
                features["vp_gex_confluence_poc"] = pd.to_numeric(df["vp_gex_confluence_poc"], errors="coerce").fillna(
                    0.0
                )

        if features.empty:
            return None
        return features.astype(float)

    def fit(self, df_train: pd.DataFrame, verbose: bool = True) -> "RLMMarkovSwitching":
        endog = self.prepare_endog(df_train)
        exog = self._prepare_features(df_train)
        model = MarkovRegression(
            endog=endog,
            exog=exog,
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
        exog = self._prepare_features(df)
        model = MarkovRegression(
            endog=endog,
            exog=exog,
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

    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic :math:`P(S_{t+1}=j \\mid S_t=i)` in statsmodels regime index order."""
        if self.fit_result is None:
            raise RuntimeError("Markov-switching model is not fitted")
        t = np.asarray(
            self.fit_result.model.regime_transition_matrix(self.fit_result.params),
            dtype=np.float64,
        )
        t = np.squeeze(t)
        if t.ndim != 2:
            raise ValueError(f"Expected 2D transition matrix, got shape {t.shape}")
        t = np.clip(t, 1e-12, 1.0)
        row = t.sum(axis=1, keepdims=True)
        row = np.where(row > 0.0, row, 1.0)
        return (t / row).astype(np.float64)

    def calibrated_transition_matrix(self, pseudocount: float | None = None) -> np.ndarray:
        t = self.transition_matrix().copy()
        alpha = float(self.config.transition_pseudocount if pseudocount is None else pseudocount)
        if alpha > 0.0:
            t = t + alpha
        t = t / t.sum(axis=1, keepdims=True)
        return np.clip(t, 1e-12, 1.0)

    def annotate(self, df: pd.DataFrame, prefix: str = "markov") -> pd.DataFrame:
        probs = self.filter(df)
        out = df.copy()
        out[f"{prefix}_probs"] = probs.to_numpy().tolist()
        out[f"{prefix}_state"] = probs.to_numpy().argmax(axis=1).astype(int)
        out[f"{prefix}_confidence"] = probs.max(axis=1).astype(float).to_numpy()
        if self.state_labels:
            out[f"{prefix}_state_label"] = [self.state_labels[int(s)] for s in out[f"{prefix}_state"]]
        return out

    def save(self, path: Path | None = None) -> None:
        path = path or self.config.model_path or Path("models/rlm_markov_switching.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RLMMarkovSwitching":
        return joblib.load(path)
