"""Kronos foundation-model integration for RLM.

Public API:
    KronosConfig           -- pydantic config loaded from configs/default.yaml
    RLMKronosPredictor     -- bar-format wrapper with caching + multi-sample paths
    KronosRegimeConfidence -- regime agreement / confidence engine
    KronosForecastPipeline -- drop-in ForecastPipeline producing mu/sigma/bands
"""

from rlm.kronos.config import KronosConfig
from rlm.kronos.forecast import KronosForecastPipeline
from rlm.kronos.predictor import RLMKronosPredictor
from rlm.kronos.regime_confidence import KronosRegimeConfidence

__all__ = [
    "KronosConfig",
    "KronosForecastPipeline",
    "KronosRegimeConfidence",
    "RLMKronosPredictor",
]
