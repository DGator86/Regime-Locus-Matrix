from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from rlm.features.scoring.regime_model import row_to_feature_view

STRATEGY_NAMES: tuple[str, ...] = (
    "bull_call_spread",
    "bear_put_spread",
    "iron_condor",
    "calendar_spread",
    "debit_spread",
    "no_trade",
)


@dataclass(frozen=True)
class StrategyScores:
    scores: dict[str, float]

    @property
    def best_strategy(self) -> str:
        return max(self.scores.items(), key=lambda item: item[1])[0]


class StrategyValueModel:
    def __init__(self, strategies: Sequence[str] = STRATEGY_NAMES) -> None:
        self.strategies = tuple(strategies)
        self._coef = np.zeros((13, len(self.strategies)), dtype=float)

    @classmethod
    def with_bootstrap_coefficients(cls) -> "StrategyValueModel":
        model = cls()
        coef = np.zeros((13, len(model.strategies)), dtype=float)

        def set_terms(strategy: str, terms: dict[int, float]) -> None:
            idx = model.strategies.index(strategy)
            for col, val in terms.items():
                coef[col, idx] = val

        # psi = [1,d,v,l,g,trend,align,delta,r,d*g,v*l,align^2,r^2]
        set_terms("bull_call_spread", {0: 0.0, 1: 1.2, 3: 0.2, 6: 0.5, 8: -0.2})
        set_terms("bear_put_spread", {0: 0.0, 1: -1.2, 3: 0.2, 6: 0.5, 8: -0.2})
        set_terms("iron_condor", {0: 0.2, 2: -0.8, 5: -0.7, 3: 0.2, 8: -0.3})
        set_terms("calendar_spread", {0: -0.1, 2: 0.2, 8: 1.0, 11: -0.02, 12: 0.03})
        set_terms("debit_spread", {0: -0.2, 2: 1.0, 5: 0.4, 6: -0.2, 8: 0.2})
        set_terms("no_trade", {0: -0.1, 3: -0.8, 8: 0.5, 12: 0.02})
        model._coef = coef
        return model

    def fit(
        self, X: pd.DataFrame | np.ndarray, Y: pd.DataFrame | np.ndarray, l2: float = 1e-3
    ) -> "StrategyValueModel":
        x_mat = self._to_design_matrix(X)
        y_mat = self._to_target_matrix(Y)
        if x_mat.shape[0] != y_mat.shape[0]:
            raise ValueError("X and Y must contain the same number of rows.")
        eye = np.eye(x_mat.shape[1], dtype=float)
        self._coef = np.linalg.solve(x_mat.T @ x_mat + l2 * eye, x_mat.T @ y_mat)
        return self

    def predict_expected_values(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        x_mat = self._to_design_matrix(X)
        return x_mat @ self._coef

    def score_row(self, row: Mapping[str, float]) -> StrategyScores:
        x_row = self._to_design_matrix(pd.DataFrame([row]))
        scores = (x_row @ self._coef)[0]
        return StrategyScores(
            scores={self.strategies[i]: float(scores[i]) for i in range(len(self.strategies))}
        )

    def _to_target_matrix(self, Y: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(Y, pd.DataFrame):
            missing = [s for s in self.strategies if s not in Y.columns]
            if missing:
                raise ValueError(f"Missing strategy columns in Y: {missing}")
            return Y.loc[:, self.strategies].to_numpy(dtype=float)
        arr = np.asarray(Y, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Y must be 2D.")
        if arr.shape[1] != len(self.strategies):
            raise ValueError(f"Expected {len(self.strategies)} target columns, got {arr.shape[1]}")
        return arr

    def _to_design_matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            psi_rows = []
            for row in X.to_dict(orient="records"):
                view = row_to_feature_view(row)
                psi_rows.append(
                    np.array(
                        [
                            1.0,
                            view.d,
                            view.v,
                            view.liquidity,
                            view.g,
                            view.trend,
                            view.align,
                            view.delta,
                            view.r_trans,
                            view.d * view.g,
                            view.v * view.liquidity,
                            view.align * view.align,
                            view.r_trans * view.r_trans,
                        ],
                        dtype=float,
                    )
                )
            return np.vstack(psi_rows) if psi_rows else np.empty((0, 13), dtype=float)

        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if arr.shape[1] != 13:
            raise ValueError("Expected engineered design matrix with 13 columns.")
        return arr
