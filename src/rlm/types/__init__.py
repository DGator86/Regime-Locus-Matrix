from rlm.forecasting.hmm import HMMConfig
from rlm.types.coordinates import MarketCoordinate
from rlm.types.forecast import ForecastConfig, ForecastSnapshot

RegimeKey = str
RLMConfig = ForecastConfig

__all__ = [
    "ForecastConfig",
    "ForecastSnapshot",
    "HMMConfig",
    "MarketCoordinate",
    "RegimeKey",
    "RLMConfig",
]
