from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    drift_gamma_alpha: float = 0.65
    sigma_floor: float = 1e-4
    direction_neutral_threshold: float = 0.3


@dataclass(frozen=True)
class ForecastSnapshot:
    mu: float
    sigma: float
    mean_price: float
    lower_1s: float
    upper_1s: float
    lower_2s: float
    upper_2s: float
