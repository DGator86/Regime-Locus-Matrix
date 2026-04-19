from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rlm.features.scoring.regime_model import RegimeModel
from rlm.features.scoring.regime_smoother import smooth_regime_probabilities
from rlm.training.sequence_features import add_sequence_features


@dataclass
class TemporalRegimeModel:
    base_model: RegimeModel
    window: int = 5
    smoothing_alpha: float = 0.25

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seq = add_sequence_features(X, window=self.window)
        design = _select_temporal_regime_features(seq, window=self.window)
        raw = self.base_model.predict_proba(design)
        return smooth_regime_probabilities(raw, alpha=self.smoothing_alpha)

    def predict(self, X: pd.DataFrame) -> list[str]:
        probs = self.predict_proba(X)
        argmax = probs.argmax(axis=1)
        return [self.base_model.labels[i] for i in argmax]


def _select_temporal_regime_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    keep = [
        "M_D",
        "M_V",
        "M_L",
        "M_G",
        "M_trend_strength",
        "M_dealer_control",
        "M_alignment",
        "M_delta_neutral",
        "M_R_trans",
        f"M_D_mean_{window}",
        f"M_V_mean_{window}",
        f"M_L_mean_{window}",
        f"M_G_mean_{window}",
        f"M_alignment_std_{window}",
        f"M_delta_neutral_std_{window}",
        f"M_R_trans_mean_{window}",
        f"M_D_slope_{window}",
        f"M_V_slope_{window}",
        f"M_G_slope_{window}",
    ]
    return df.loc[:, [c for c in keep if c in df.columns]].copy()
