from __future__ import annotations

import numpy as np
import pandas as pd

from rlm.standardization.transforms import sigma_floor
from rlm.types.forecast import ForecastConfig


def compute_baseline_move_scale(
    close: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Baseline drift magnitude scale:
    rolling median absolute return.
    """
    returns = close.pct_change()
    return returns.abs().rolling(window=window, min_periods=max(10, window // 3)).median()


def compute_baseline_vol_scale(
    close: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Baseline volatility scale:
    rolling median absolute deviation around rolling median return.
    """
    returns = close.pct_change()

    rolling_med = returns.rolling(window=window, min_periods=max(10, window // 3)).median()

    def _mad(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        med = np.nanmedian(x)
        return float(np.nanmedian(np.abs(x - med)))

    mad = returns.rolling(window=window, min_periods=max(10, window // 3)).apply(
        _mad,
        raw=True,
    )

    # fallback if rolling MAD gets too sparse early
    fallback = returns.abs().rolling(window=window, min_periods=max(10, window // 3)).median()
    return mad.fillna(fallback).clip(lower=0.0)


def compute_mu(
    s_d: pd.Series,
    s_g: pd.Series,
    b_m: pd.Series,
    config: ForecastConfig | None = None,
) -> pd.Series:
    """
    mu = S_D * b_m * (1 + alpha * S_G)
    with mu = 0 when |S_D| < neutral threshold
    """
    cfg = config or ForecastConfig()

    s_g = s_g.fillna(0.0)
    mu = s_d * b_m * (1.0 + cfg.drift_gamma_alpha * s_g)
    mu = mu.where(s_d.abs() >= cfg.direction_neutral_threshold, 0.0)
    return mu


def compute_sigma(
    s_v: pd.Series,
    s_l: pd.Series,
    s_g: pd.Series,
    b_sigma: pd.Series,
    config: ForecastConfig | None = None,
) -> pd.Series:
    """
    sigma = max(
        sigma_floor,
        b_sigma * (0.5 + 0.5*S_V) * (1.4 - 0.7*S_L) * (1.15 - 0.45*S_G)
    )
    """
    cfg = config or ForecastConfig()

    s_g = s_g.fillna(0.0)
    raw_sigma = (
        b_sigma
        * (0.5 + 0.5 * s_v)
        * (1.4 - 0.7 * s_l)
        * (1.15 - 0.45 * s_g)
    )

    return raw_sigma.apply(lambda x: sigma_floor(x, cfg.sigma_floor))


def estimate_distribution(
    df: pd.DataFrame,
    config: ForecastConfig | None = None,
    move_window: int = 100,
    vol_window: int = 100,
) -> pd.DataFrame:
    """
    Requires:
      close, S_D, S_V, S_L, S_G
    Returns original df + b_m, b_sigma, mu, sigma, mean_price.
    """
    cfg = config or ForecastConfig()

    required = ["close", "S_D", "S_V", "S_L", "S_G"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for distribution estimation: {missing}")

    out = df.copy()

    out["b_m"] = compute_baseline_move_scale(out["close"], window=move_window)
    out["b_sigma"] = compute_baseline_vol_scale(out["close"], window=vol_window)

    out["mu"] = compute_mu(out["S_D"], out["S_G"], out["b_m"], cfg)
    out["sigma"] = compute_sigma(out["S_V"], out["S_L"], out["S_G"], out["b_sigma"], cfg)
    out["mean_price"] = out["close"] * (1.0 + out["mu"])

    return out
