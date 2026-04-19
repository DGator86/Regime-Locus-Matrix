from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rlm.roee.strategy_value_model import StrategyValueModel
from rlm.training.sequence_features import add_sequence_features


@dataclass
class TemporalStrategyValueModel:
    base_model: StrategyValueModel
    window: int = 5

    @property
    def strategies(self) -> tuple[str, ...]:
        return self.base_model.strategies

    def predict_expected_values(self, X: pd.DataFrame) -> np.ndarray:
        seq = add_sequence_features(X, window=self.window)
        design = _select_temporal_value_features(seq, window=self.window)
        return self.base_model.predict_expected_values(design)


def _select_temporal_value_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    keep = [
        "M_D",
        "M_V",
        "M_L",
        "M_G",
        "M_trend_strength",
        "M_alignment",
        "M_delta_neutral",
        "M_R_trans",
        f"M_D_mean_{window}",
        f"M_V_mean_{window}",
        f"M_L_mean_{window}",
        f"M_alignment_mean_{window}",
        f"M_alignment_std_{window}",
        f"M_delta_neutral_mean_{window}",
        f"M_R_trans_mean_{window}",
        f"M_D_slope_{window}",
        f"M_V_slope_{window}",
    ]
    return df.loc[:, [c for c in keep if c in df.columns]].copy()
