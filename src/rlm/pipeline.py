"""Backward-compatibility re-export. Canonical locations: ``rlm.core.pipeline`` and ``rlm.ingestion.pipeline``."""

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.ingestion.pipeline import IngestionPipeline

__all__ = ["FullRLMConfig", "FullRLMPipeline", "PipelineResult", "IngestionPipeline"]
