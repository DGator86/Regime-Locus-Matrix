from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.forecasting.markov_switching import (
    MarkovSwitchingConfig,
    RLMMarkovSwitching,
)

ProbSource = Literal["regime_probs", "hmm_probs", "markov_probs", "probs"]


class MultiTimeframeRegimeModel:
    """Blend low-timeframe regime probabilities with higher-timeframe priors.

    The low-timeframe (LTF) probabilities come from a fitted HMM or Markov-switching
    model on the current feature frame. Higher-timeframe (HTF) probabilities are loaded
    from precomputed parquet files and aligned backward-in-time per row timestamp.
    """

    def __init__(
        self,
        *,
        model: Literal["hmm", "markov"] = "hmm",
        hmm_config: HMMConfig | None = None,
        markov_config: MarkovSwitchingConfig | None = None,
        htf_prob_paths: dict[str, Path | str] | None = None,
        htf_weights: dict[str, float] | None = None,
        ltf_weight: float = 0.7,
        timestamp_col: str = "timestamp",
    ) -> None:
        self.model_name = model
        self.hmm = RLMHMM(hmm_config or HMMConfig()) if model == "hmm" else None
        self.markov = (
            RLMMarkovSwitching(markov_config or MarkovSwitchingConfig())
            if model == "markov"
            else None
        )
        self.htf_prob_paths = {k: Path(v) for k, v in (htf_prob_paths or {}).items()}
        self.htf_weights = htf_weights or {}
        self.ltf_weight = float(ltf_weight)
        self.timestamp_col = timestamp_col

        if self.ltf_weight < 0.0 or self.ltf_weight > 1.0:
            raise ValueError("ltf_weight must be within [0.0, 1.0].")
        if not self.htf_prob_paths:
            raise ValueError("htf_prob_paths must include at least one timeframe parquet path.")

    def fit(self, df_train: pd.DataFrame, verbose: bool = False) -> "MultiTimeframeRegimeModel":
        if self.hmm is not None:
            self.hmm.fit(df_train, verbose=verbose)
        elif self.markov is not None:
            self.markov.fit(df_train, verbose=verbose)
        else:
            raise RuntimeError("No LTF model configured.")
        return self

    def annotate(self, df: pd.DataFrame, *, prefix: str = "mtf") -> pd.DataFrame:
        ltf_probs = self._ltf_probs(df)
        htf_probs = self._aligned_htf_probs(df)
        blended = self._blend(ltf_probs, htf_probs)

        out = df.copy()
        out[f"{prefix}_probs"] = blended.tolist()
        out[f"{prefix}_state"] = blended.argmax(axis=1).astype(int)
        out[f"{prefix}_confidence"] = blended.max(axis=1).astype(float)

        labels = self._state_labels()
        if labels:
            out[f"{prefix}_state_label"] = [labels[int(s)] for s in out[f"{prefix}_state"]]
        return out

    def _ltf_probs(self, df: pd.DataFrame) -> np.ndarray:
        if self.hmm is not None:
            return self.hmm.predict_proba_filtered(df)
        if self.markov is not None:
            return self.markov.filter(df).to_numpy(dtype=float)
        raise RuntimeError("No LTF model configured.")

    def _state_labels(self) -> list[str] | None:
        if self.hmm is not None:
            return self.hmm.state_labels
        if self.markov is not None:
            return self.markov.state_labels
        return None

    def _aligned_htf_probs(self, df: pd.DataFrame) -> np.ndarray:
        timestamps = self._frame_timestamps(df)
        weighted: list[np.ndarray] = []
        weights: list[float] = []
        for timeframe, path in self.htf_prob_paths.items():
            if not path.is_file():
                continue
            probs_df = pd.read_parquet(path)
            parsed = self._parse_probs_frame(probs_df)
            if parsed.empty:
                continue
            aligned = pd.merge_asof(
                pd.DataFrame({"_ts": timestamps}).sort_values("_ts"),
                parsed.sort_values("_ts"),
                on="_ts",
                direction="backward",
            )
            arr = np.vstack(aligned["_probs"].apply(self._to_prob_vector).to_list())
            weighted.append(arr)
            weights.append(float(self.htf_weights.get(timeframe, 1.0)))

        if not weighted:
            raise ValueError("No valid HTF probability parquet sources were found.")

        total_w = float(np.sum(weights))
        if total_w <= 0:
            raise ValueError("Sum of HTF weights must be positive.")

        n_states = min(arr.shape[1] for arr in weighted)
        mix = np.zeros((len(df), n_states), dtype=float)
        for arr, w in zip(weighted, weights, strict=False):
            mix += (w / total_w) * arr[:, :n_states]
        return self._normalize(mix)

    def _blend(self, ltf_probs: np.ndarray, htf_probs: np.ndarray) -> np.ndarray:
        n_states = min(ltf_probs.shape[1], htf_probs.shape[1])
        ltf = self._normalize(ltf_probs[:, :n_states])
        htf = self._normalize(htf_probs[:, :n_states])
        blended = (self.ltf_weight * ltf) + ((1.0 - self.ltf_weight) * htf)
        return self._normalize(blended)

    def _frame_timestamps(self, df: pd.DataFrame) -> pd.Series:
        if self.timestamp_col in df.columns:
            ts = pd.to_datetime(df[self.timestamp_col], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(pd.Index(df.index), utc=True, errors="coerce")
        return ts.fillna(method="ffill").fillna(method="bfill")

    @staticmethod
    def _parse_probs_frame(df: pd.DataFrame) -> pd.DataFrame:
        cols = set(df.columns)
        src_col: ProbSource | None = None
        for candidate in ("regime_probs", "hmm_probs", "markov_probs", "probs"):
            if candidate in cols:
                src_col = candidate
                break
        if src_col is None:
            return pd.DataFrame(columns=["_ts", "_probs"])

        if "timestamp" in cols:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(pd.Index(df.index), utc=True, errors="coerce")

        out = pd.DataFrame({"_ts": ts, "_probs": df[src_col]})
        out = out.dropna(subset=["_ts"]).sort_values("_ts")
        out["_probs"] = out["_probs"].apply(MultiTimeframeRegimeModel._to_prob_vector)
        return out

    @staticmethod
    def _to_prob_vector(value: object) -> np.ndarray:
        if isinstance(value, np.ndarray):
            arr = value.astype(float)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=float)
        elif isinstance(value, str):
            txt = value.strip().removeprefix("[").removesuffix("]")
            if not txt:
                arr = np.array([1.0], dtype=float)
            else:
                arr = np.asarray([float(part) for part in txt.split(",")], dtype=float)
        else:
            arr = np.asarray([float(value)], dtype=float)
        return MultiTimeframeRegimeModel._normalize(arr.reshape(1, -1)).ravel()

    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = arr.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            arr[zero_rows] = 1.0 / max(arr.shape[1], 1)
            row_sums = arr.sum(axis=1, keepdims=True)
        return arr / row_sums
