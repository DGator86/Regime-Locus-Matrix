from typing import Any

from rlm.forecasting.hmm import HMMConfig, RLMHMM
from rlm.forecasting.kronos_forecast import KronosConfig, KronosForecastPipeline
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridKronosForecastPipeline,
)

__all__ = [
    "ForecastPipeline",
    "HybridForecastPipeline",
    "HybridKronosForecastPipeline",
    "KronosConfig",
    "KronosForecastPipeline",
    "RLMHMM",
    "HMMConfig",
    "LiveRegimeModelConfig",
    "load_live_regime_model",
    "save_live_regime_model",
]


def __getattr__(name: str) -> Any:
    if name in {"LiveRegimeModelConfig", "load_live_regime_model", "save_live_regime_model"}:
        from rlm.forecasting.live_model import (
            LiveRegimeModelConfig,
            load_live_regime_model,
            save_live_regime_model,
        )

        exports = {
            "LiveRegimeModelConfig": LiveRegimeModelConfig,
            "load_live_regime_model": load_live_regime_model,
            "save_live_regime_model": save_live_regime_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
