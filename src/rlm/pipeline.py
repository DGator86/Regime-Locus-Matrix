"""Backward-compatibility re-export. Canonical location: rlm.core.pipeline.  (PR #41)"""

from rlm.core.pipeline import FullRLMConfig, FullRLMPipeline, PipelineResult

__all__ = ["FullRLMConfig", "FullRLMPipeline", "PipelineResult"]
