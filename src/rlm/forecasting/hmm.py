"""HMM overlay for RLM: uses standardized RLM scores as observations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from pydantic import BaseModel, Field
from scipy.special import logsumexp, softmax

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration path
    njit = None

try:
    from numba import cuda as numba_cuda  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - GPU path is optional
    numba_cuda = None


def _forward_filter_logspace_numpy(
    log_frame: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    n_samples, n_components = log_frame.shape
    log_alpha = np.zeros((n_samples, n_components), dtype=np.float64)
    log_alpha[0] = log_start + log_frame[0]
    for t in range(1, n_samples):
        log_alpha[t] = logsumexp(log_trans + log_alpha[t - 1][:, np.newaxis], axis=0) + log_frame[t]
    return log_alpha


if njit is not None:

    @njit(cache=True)
    def _row_logsumexp(values: np.ndarray) -> float:
        max_value = np.max(values)
        acc = 0.0
        for i in range(values.shape[0]):
            acc += np.exp(values[i] - max_value)
        return float(max_value + np.log(acc))


    @njit(cache=True)
    def _forward_filter_logspace_numba(
        log_frame: np.ndarray,
        log_start: np.ndarray,
        log_trans: np.ndarray,
    ) -> np.ndarray:
        n_samples, n_components = log_frame.shape
        log_alpha = np.zeros((n_samples, n_components), dtype=np.float64)
        log_alpha[0] = log_start + log_frame[0]
        work = np.zeros(n_components, dtype=np.float64)
        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    work[i] = log_trans[i, j] + log_alpha[t - 1, i]
                log_alpha[t, j] = _row_logsumexp(work) + log_frame[t, j]
        return log_alpha

else:

    def _forward_filter_logspace_numba(
        log_frame: np.ndarray,
        log_start: np.ndarray,
        log_trans: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("Numba backend requested but numba is unavailable.")


class HMMConfig(BaseModel):
    n_states: int = Field(
        6,
        ge=4,
        le=12,
        description="Number of hidden states (empirically 6 works well)",
    )
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    n_iter: int = 100
    random_state: int = 42
    model_path: Path | None = None
    filter_backend: Literal["auto", "numpy", "numba"] = "auto"
    prefer_gpu: bool = Field(
        False,
        description="Request GPU acceleration metadata when a CUDA runtime is available.",
    )


class RLMHMM:
    def __init__(self, config: HMMConfig = HMMConfig()):
        self.config = config
        self.model: hmm.GaussianHMM | None = None
        self.state_labels: list[str] | None = None
        self.last_filter_backend: str | None = None

    def prepare_observations(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 4) array of standardized scores."""
        required = ["S_D", "S_V", "S_L", "S_G"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for HMM observations: {missing}")

        block = df[list(required)].apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
        scores = np.column_stack(
            [
                block["S_D"].values,
                block["S_V"].values,
                block["S_L"].values,
                block["S_G"].values,
            ]
        )
        return np.clip(np.nan_to_num(scores, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)

    def fit(self, df_train: pd.DataFrame, verbose: bool = True) -> "RLMHMM":
        obs = self.prepare_observations(df_train)
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
            init_params="stmc",
        )
        self.model.fit(obs)
        if verbose:
            print(
                f"HMM fitted with {self.config.n_states} states, "
                f"log-likelihood: {self.model.score(obs):.2f}"
            )
        self._auto_label_states(df_train)
        return self

    def _auto_label_states(self, df: pd.DataFrame) -> None:
        if self.model is None:
            raise RuntimeError("HMM model must be fitted before auto-labeling states")
        obs = self.prepare_observations(df)
        states = self.model.predict(obs)
        labels: list[str] = []
        for s in range(self.config.n_states):
            mask = states == s
            if mask.sum() == 0:
                labels.append(f"state_{s}")
                continue

            if "regime_key" in df.columns and not df.loc[mask, "regime_key"].dropna().empty:
                most_common_key = str(df.loc[mask, "regime_key"].mode().iloc[0])
                labels.append(f"{most_common_key}_like")
            else:
                labels.append(f"state_{s}")

        self.state_labels = labels

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("HMM model is not fitted")
        obs = self.prepare_observations(df)
        return self.model.predict_proba(obs)

    def predict_proba_filtered(self, df: pd.DataFrame) -> np.ndarray:
        """Forward (filtering) state probabilities P(z_t | x_{1:t}) only — no future observations.

        Use this for walk-forward / live paths. :meth:`predict_proba` uses forward-backward
        smoothing and conditions each row on the **entire** sequence (lookahead within ``df``).
        """
        if self.model is None:
            raise RuntimeError("HMM model is not fitted")
        obs = self.prepare_observations(df)
        model = self.model
        log_frame = model._compute_log_likelihood(obs)
        n_samples, n_components = log_frame.shape
        log_start = np.log(model.startprob_ + 1e-300).astype(np.float64)
        log_trans = np.log(model.transmat_ + 1e-300).astype(np.float64)
        backend = self._resolve_filter_backend()
        if backend == "numba":
            log_alpha = _forward_filter_logspace_numba(log_frame.astype(np.float64), log_start, log_trans)
        else:
            log_alpha = _forward_filter_logspace_numpy(log_frame.astype(np.float64), log_start, log_trans)
        out = np.zeros((n_samples, n_components), dtype=np.float64)
        for t in range(n_samples):
            out[t] = softmax(log_alpha[t])
        gpu_requested = bool(self.config.prefer_gpu)
        gpu_suffix = ""
        if gpu_requested:
            gpu_suffix = "+gpu" if self._gpu_runtime_available() else "+gpu-fallback"
        self.last_filter_backend = f"{backend}{gpu_suffix}"
        return out

    def most_likely_state(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("HMM model is not fitted")
        obs = self.prepare_observations(df)
        return self.model.predict(obs)

    def most_likely_state_filtered(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba_filtered(df)
        return np.argmax(probs, axis=1).astype(int)

    def save(self, path: Path | None = None) -> None:
        path = path or self.config.model_path or Path("models/rlm_hmm.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RLMHMM":
        return joblib.load(path)

    def _resolve_filter_backend(self) -> Literal["numpy", "numba"]:
        requested = self.config.filter_backend
        if requested == "numpy":
            return "numpy"
        if requested == "numba":
            if njit is None:
                raise RuntimeError("HMM numba backend requested but numba is unavailable.")
            return "numba"
        return "numba" if njit is not None else "numpy"

    @staticmethod
    def _gpu_runtime_available() -> bool:
        if numba_cuda is None:
            return False
        try:
            return bool(numba_cuda.is_available())
        except Exception:  # pragma: no cover - CUDA probing is environment-specific
            return False
