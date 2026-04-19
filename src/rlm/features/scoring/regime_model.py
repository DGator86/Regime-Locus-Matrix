from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from rlm.training.artifacts import RegimeModelArtifact

RegimeLabel = str

REGIME_LABELS: tuple[RegimeLabel, ...] = (
    "trend_up_stable",
    "trend_down_stable",
    "range_compression",
    "mean_reversion",
    "breakout_expansion",
    "transition",
    "chaos",
    "no_trade",
)

_STATE_COLUMNS: tuple[str, ...] = (
    "M_D",
    "M_V",
    "M_L",
    "M_G",
    "M_trend_strength",
    "M_dealer_control",
    "M_alignment",
    "M_delta_neutral",
    "M_R_trans",
)


@dataclass(frozen=True)
class FeatureView:
    d: float
    v: float
    liquidity: float
    g: float
    trend: float
    dealer: float
    align: float
    delta: float
    r_trans: float


def _safe_float(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def row_to_feature_view(row: Mapping[str, float]) -> FeatureView:
    d_raw = _safe_float(row, "M_D", np.nan)
    v_raw = _safe_float(row, "M_V", np.nan)
    l_raw = _safe_float(row, "M_L", np.nan)
    g_raw = _safe_float(row, "M_G", np.nan)
    if not np.isfinite([d_raw, v_raw, l_raw, g_raw]).all():
        return FeatureView(0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0)

    d = d_raw - 5.0
    v = v_raw - 5.0
    liquidity = l_raw - 5.0
    g = g_raw - 5.0
    trend = _safe_float(row, "M_trend_strength", abs(d))
    dealer = _safe_float(row, "M_dealer_control", abs(g))
    align = _safe_float(row, "M_alignment", d * g)
    delta = _safe_float(row, "M_delta_neutral", np.sqrt(d**2 + v**2 + liquidity**2 + g**2))
    r_trans = _safe_float(row, "M_R_trans", 0.0)
    return FeatureView(d, v, liquidity, g, trend, dealer, align, delta, r_trans)


def _phi(view: FeatureView) -> np.ndarray:
    return np.array(
        [
            1.0,
            view.d,
            view.v,
            view.liquidity,
            view.g,
            view.trend,
            view.dealer,
            view.align,
            view.delta,
            view.r_trans,
            view.d * view.g,
            view.v * view.liquidity,
            view.align * view.align,
            view.r_trans * view.r_trans,
            view.liquidity * view.liquidity,
        ],
        dtype=float,
    )


class RegimeModel:
    def __init__(self, labels: Sequence[str] = REGIME_LABELS) -> None:
        self.labels = tuple(labels)
        self._label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self._weights = np.zeros((len(self.labels), 15), dtype=float)

    @classmethod
    def with_bootstrap_coefficients(cls) -> "RegimeModel":
        model = cls()
        w = np.zeros((len(model.labels), 15), dtype=float)

        def set_terms(label: str, terms: dict[int, float]) -> None:
            idx = model._label_to_idx[label]
            for col, val in terms.items():
                w[idx, col] = val

        set_terms("trend_up_stable", {0: 0.0, 1: 1.3, 2: -0.4, 3: 0.3, 7: 0.8, 9: -0.2})
        set_terms("trend_down_stable", {0: 0.0, 1: -1.3, 2: -0.4, 3: 0.3, 7: 0.8, 9: -0.2})
        set_terms("range_compression", {0: 0.4, 2: -0.8, 4: 0.5, 5: -0.8, 3: 0.2, 9: -0.3})
        set_terms("mean_reversion", {0: 0.2, 7: -1.1, 2: 0.2, 3: 0.2})
        set_terms("breakout_expansion", {0: -0.3, 2: 1.0, 5: 0.5, 7: -0.3, 9: 0.2})
        set_terms("transition", {0: -0.5, 2: 0.2, 3: -0.2, 8: 0.2, 9: 1.0, 13: 0.05})
        set_terms("chaos", {0: -0.7, 2: 1.1, 3: -0.9, 9: 0.3})
        set_terms("no_trade", {0: -0.2, 3: -0.7, 5: -0.4, 8: -0.2, 9: 0.2, 14: 0.05})

        model._weights = w
        return model

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Sequence[str],
        *,
        learning_rate: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-4,
    ) -> "RegimeModel":
        X_mat = self._to_feature_matrix(X)
        return self.fit_design_matrix(
            X_mat,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
        )

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_mat = self._to_feature_matrix(X)
        return self.predict_proba_design_matrix(X_mat)

    def predict(self, X: pd.DataFrame | np.ndarray) -> list[str]:
        probs = self.predict_proba(X)
        argmax = probs.argmax(axis=1)
        return [self.labels[idx] for idx in argmax]

    def fit_design_matrix(
        self,
        X_design: np.ndarray,
        y: Sequence[str],
        *,
        learning_rate: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-4,
    ) -> "RegimeModel":
        arr = np.asarray(X_design, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X_design must be 2D")
        if arr.shape[0] != len(y):
            raise ValueError("X_design and y length mismatch")
        y_idx = np.array([self._label_to_idx[str(label)] for label in y], dtype=int)
        one_hot = np.eye(len(self.labels), dtype=float)[y_idx]
        self._weights = np.zeros((len(self.labels), arr.shape[1]), dtype=float)
        n = float(arr.shape[0])
        for _ in range(max(epochs, 1)):
            logits = arr @ self._weights.T
            probs = _softmax(logits)
            grad = ((probs - one_hot).T @ arr) / n + l2 * self._weights
            self._weights -= learning_rate * grad
        return self

    def predict_proba_design_matrix(self, X_design: np.ndarray) -> np.ndarray:
        arr = np.asarray(X_design, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X_design must be 2D")
        logits = arr @ self._weights.T
        return _softmax(logits)

    @classmethod
    def from_artifact(cls, artifact: RegimeModelArtifact) -> "RegimeModel":
        model = cls(labels=artifact.labels)
        model._weights = np.asarray(artifact.weights, dtype=float)
        return model

    def to_artifact(
        self,
        *,
        trained_at: str,
        training_rows: int,
        source_symbols: list[str],
        feature_names: list[str],
        target_mode: str | None = None,
        label_mode: str | None = None,
        horizon: int | None = None,
        training_start: str | None = None,
        training_end: str | None = None,
        benchmark_summary: dict[str, float] | None = None,
        simulator_version: str | None = None,
        execution_model_version: str | None = None,
        train_split: float | None = None,
        validation_rows: int | None = None,
        sequence_window: int | None = None,
        smoothing_alpha: float | None = None,
        temporal_model: bool = False,
        validation_matrix_summary: dict[str, float] | None = None,
        feature_ablation_summary: dict[str, float] | None = None,
    ) -> RegimeModelArtifact:
        return RegimeModelArtifact(
            labels=list(self.labels),
            weights=self._weights.tolist(),
            feature_names=feature_names,
            trained_at=trained_at,
            training_rows=training_rows,
            source_symbols=source_symbols,
            target_mode=target_mode,
            label_mode=label_mode,
            horizon=horizon,
            training_start=training_start,
            training_end=training_end,
            benchmark_summary=benchmark_summary,
            simulator_version=simulator_version,
            execution_model_version=execution_model_version,
            train_split=train_split,
            validation_rows=validation_rows,
            sequence_window=sequence_window,
            smoothing_alpha=smoothing_alpha,
            temporal_model=temporal_model,
            validation_matrix_summary=validation_matrix_summary,
            feature_ablation_summary=feature_ablation_summary,
        )

    def _to_feature_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            views = [row_to_feature_view(row) for row in X.to_dict(orient="records")]
            return np.vstack([_phi(v) for v in views]) if views else np.empty((0, 15), dtype=float)

        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if arr.shape[1] == 15:
            return arr
        if arr.shape[1] != len(_STATE_COLUMNS):
            raise ValueError(
                "Expected "
                f"{len(_STATE_COLUMNS)} state columns or 15 engineered features; "
                f"got {arr.shape[1]}."
            )
        views = [FeatureView(*row.tolist()) for row in arr]
        return np.vstack([_phi(v) for v in views])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
