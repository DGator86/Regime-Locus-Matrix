from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    drift_gamma_alpha: float = 0.65
    sigma_floor: float = 1e-4
    direction_neutral_threshold: float = 0.3
    realized_vol_window: int = 20
    realized_vol_annualization: float = 252.0
    probabilistic_lower_quantile: float = 0.1
    probabilistic_upper_quantile: float = 0.9


@dataclass(frozen=True)
class ForecastSnapshot:
    mu: float
    sigma: float
    mean_price: float
    lower_1s: float
    upper_1s: float
    lower_2s: float
    upper_2s: float
    forecast_return: float | None = None
    forecast_return_lower: float | None = None
    forecast_return_median: float | None = None
    forecast_return_upper: float | None = None
    forecast_uncertainty: float | None = None
    realized_vol: float | None = None
