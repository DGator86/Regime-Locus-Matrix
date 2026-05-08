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
from rlm.forecasting.probabilistic_regime_engine import (
    PREConfig,
    ProbabilisticRegimeEngine,
    ProbabilisticRegimeEngineMTF,
    RegimeSignal,
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
    "PREConfig",
    "ProbabilisticRegimeEngine",
    "ProbabilisticRegimeEngineMTF",
    "RegimeSignal",
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
