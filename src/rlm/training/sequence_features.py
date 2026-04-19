from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COORD_COLS: list[str] = [
    "M_D",
    "M_V",
    "M_L",
    "M_G",
    "M_trend_strength",
    "M_dealer_control",
    "M_alignment",
    "M_delta_neutral",
    "M_R_trans",
]


def add_sequence_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")
    out = df.copy()
    for col in BASE_COORD_COLS:
        if col not in out.columns:
            continue
        out[f"{col}_mean_{window}"] = out[col].rolling(window, min_periods=1).mean()
        out[f"{col}_std_{window}"] = out[col].rolling(window, min_periods=1).std().fillna(0.0)
        out[f"{col}_delta_1"] = out[col].diff().fillna(0.0)
        out[f"{col}_slope_{window}"] = _rolling_slope(out[col], window)
    return out


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def fit_slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        xv = x[: len(values)]
        yv = np.asarray(values, dtype=float)
        xm = xv.mean()
        ym = yv.mean()
        denom = np.sum((xv - xm) ** 2)
        if denom <= 0:
            return 0.0
        return float(np.sum((xv - xm) * (yv - ym)) / denom)

    return series.rolling(window, min_periods=2).apply(fit_slope, raw=True).fillna(0.0)
