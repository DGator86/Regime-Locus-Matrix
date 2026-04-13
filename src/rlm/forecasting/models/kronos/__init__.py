"""Kronos foundation-model integration for RLM.

Public API:
    KronosConfig           -- pydantic config loaded from configs/default.yaml
    RLMKronosPredictor     -- bar-format wrapper with caching + multi-sample paths
    KronosRegimeConfidence -- regime agreement / confidence engine
    KronosForecastPipeline -- drop-in ForecastPipeline producing mu/sigma/bands
"""

from rlm.forecasting.models.kronos.config import KronosConfig
from rlm.forecasting.models.kronos.forecast import KronosForecastPipeline
from rlm.forecasting.models.kronos.predictor import RLMKronosPredictor
from rlm.forecasting.models.kronos.regime_confidence import KronosRegimeConfidence

__all__ = [
    "KronosConfig",
    "KronosForecastPipeline",
    "KronosRegimeConfidence",
    "RLMKronosPredictor",
]
