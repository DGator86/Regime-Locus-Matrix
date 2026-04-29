"""Backward-compatibility re-exports for ``rlm.kronos.*`` import paths."""

from __future__ import annotations

from typing import Any

from rlm.forecasting.kronos_config import KronosConfig
from rlm.forecasting.models.kronos import KronosRegimeConfidence, RLMKronosPredictor

__all__ = [
    "KronosConfig",
    "KronosForecastPipeline",
    "KronosRegimeConfidence",
    "RLMKronosPredictor",
]


def __getattr__(name: str) -> Any:
    if name == "KronosForecastPipeline":
        from rlm.forecasting.kronos_forecast import KronosForecastPipeline

        return KronosForecastPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
