"""rlm.persona — lightweight four-stage trade interpretation pipeline.

Stages (in order):
    Seven  — signal normalization and structured bias interpretation
    Garak  — trap / deception / false-breakout detection
    Sisko  — final trade directive authority
    Data   — post-trade audit, empirical validation, edge tracking

Quick start::

    from rlm.core.pipeline import FullRLMPipeline
    from rlm.persona import PersonaDecisionPipeline

    result = FullRLMPipeline().run(bars_df)
    persona = PersonaDecisionPipeline().run(result)
    print(persona.sisko.directive)   # "long" | "short" | "no_trade"
    import json; print(json.dumps(persona.to_dict(), indent=2))
"""

from rlm.persona.config import PersonaConfig
from rlm.persona.models import (
    DataStageOutput,
    GarakStageOutput,
    PersonaInputs,
    PersonaPipelineResult,
    SevenStageOutput,
    SiskoStageOutput,
)
from rlm.persona.pipeline import PersonaDecisionPipeline

__all__ = [
    "PersonaConfig",
    "PersonaDecisionPipeline",
    "PersonaInputs",
    "PersonaPipelineResult",
    "SevenStageOutput",
    "GarakStageOutput",
    "SiskoStageOutput",
    "DataStageOutput",
]
