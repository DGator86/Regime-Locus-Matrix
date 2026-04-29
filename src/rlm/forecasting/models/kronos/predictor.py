"""Thin predictor wrapper — delegates to an injected ``predict_paths`` mock or torch model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rlm.forecasting.kronos_config import KronosConfig


class RLMKronosPredictor:
    """Loads the HuggingFace Kronos stack when available, or wraps a test/mock ``predict_paths``."""

    def __init__(
        self,
        config: KronosConfig | None = None,
        predictor: Any | None = None,
    ) -> None:
        self.config = config or KronosConfig.from_yaml()
        self._delegate = predictor

    def predict_paths(
        self,
        df: pd.DataFrame,
        future_timestamps: Any | None = None,
    ) -> np.ndarray:
        if self._delegate is not None:
            try:
                return np.asarray(
                    self._delegate.predict_paths(df, future_timestamps=future_timestamps),
                    dtype=float,
                )
            except TypeError:
                return np.asarray(self._delegate.predict_paths(df), dtype=float)

        raise ImportError(
            "RLMKronosPredictor needs predictor=... (tests/mocks) or a vendored Torch Kronos stack."
        )
