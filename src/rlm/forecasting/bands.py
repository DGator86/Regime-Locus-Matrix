from __future__ import annotations

import numpy as np
import pandas as pd


def compute_state_matrix_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires:
      close, mean_price, sigma

    sigma is a return-scale quantity.
    Price-space bands are:
      mean_price ± close * sigma
      mean_price ± close * 2*sigma
    """
    required = ["close", "mean_price", "sigma"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for band computation: {missing}")

    out = df.copy()

    # Floor half-width in price space so tiny sigma or penny names do not collapse bands to noise.
    half_1s = (out["close"].abs() * out["sigma"].abs()).fillna(0.0).astype(float)
    min_half = (out["close"].abs() * 1e-5).clip(lower=1e-8)
    half_1s = np.maximum(half_1s, min_half)

    out["lower_1s"] = out["mean_price"] - half_1s
    out["upper_1s"] = out["mean_price"] + half_1s

    out["lower_2s"] = out["mean_price"] - 2.0 * half_1s
    out["upper_2s"] = out["mean_price"] + 2.0 * half_1s

    out["core_zone_width"] = out["upper_1s"] - out["lower_1s"]
    out["full_zone_width_2s"] = out["upper_2s"] - out["lower_2s"]
    out["band_half_width_1s"] = half_1s

    return out


def label_price_zone(
    realized_next_price: pd.Series,
    lower_1s: pd.Series,
    upper_1s: pd.Series,
    lower_2s: pd.Series,
    upper_2s: pd.Series,
) -> pd.Series:
    """
    Labels where the realized next price landed relative to the state matrix.
    """
    zone = pd.Series(index=realized_next_price.index, dtype="object")

    zone[(realized_next_price >= lower_1s) & (realized_next_price <= upper_1s)] = "core"
    zone[
        ((realized_next_price >= lower_2s) & (realized_next_price < lower_1s))
        | ((realized_next_price > upper_1s) & (realized_next_price <= upper_2s))
    ] = "transition"
    zone[(realized_next_price < lower_2s) | (realized_next_price > upper_2s)] = "breakout"

    return zone
