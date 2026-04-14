"""Backward-compatibility re-exports for primary pipeline entry points."""

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult
from rlm.ingestion.pipeline import IngestionPipeline

__all__ = ["FullRLMConfig", "FullRLMPipeline", "PipelineResult", "IngestionPipeline"]
