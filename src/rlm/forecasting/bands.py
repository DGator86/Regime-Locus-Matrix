from __future__ import annotations

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

    price_sigma = out["close"] * out["sigma"]

    out["lower_1s"] = out["mean_price"] - price_sigma
    out["upper_1s"] = out["mean_price"] + price_sigma

    out["lower_2s"] = out["mean_price"] - 2.0 * price_sigma
    out["upper_2s"] = out["mean_price"] + 2.0 * price_sigma

    out["core_zone_width"] = out["upper_1s"] - out["lower_1s"]
    out["full_zone_width_2s"] = out["upper_2s"] - out["lower_2s"]

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
