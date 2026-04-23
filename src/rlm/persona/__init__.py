"""
Persona decision pipeline — lightweight, deterministic, typed.

Stages:
  Seven  → signal normalization / alignment scoring
  Garak  → trap / false-breakout detection
  Sisko  → final trade directive authority
  Data   → post-trade audit / edge tracking
"""

from rlm.persona.models import (
    DataStageOutput,
    GarakStageOutput,
    PersonaPipelineInput,
    PersonaPipelineResult,
    SiskoStageOutput,
    SevenStageOutput,
)
from rlm.persona.pipeline import PersonaDecisionPipeline

__all__ = [
    "PersonaPipelineInput",
    "SevenStageOutput",
    "GarakStageOutput",
    "SiskoStageOutput",
    "DataStageOutput",
    "PersonaPipelineResult",
    "PersonaDecisionPipeline",
]
