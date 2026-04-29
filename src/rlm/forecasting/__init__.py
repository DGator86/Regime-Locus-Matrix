from typing import Any

from rlm.forecasting.engines import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridKronosForecastPipeline,
)
from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.forecasting.kronos_forecast import (
    KronosBlendPipeline,
    KronosConfig,
    KronosForecastPipeline,
    apply_kronos_blend,
)

__all__ = [
    "ForecastPipeline",
    "HybridForecastPipeline",
    "HybridKronosForecastPipeline",
    "KronosBlendPipeline",
    "KronosConfig",
    "KronosForecastPipeline",
    "apply_kronos_blend",
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
