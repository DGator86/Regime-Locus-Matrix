"""Backward-compatibility re-exports for primary pipeline entry points."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["FullRLMConfig", "FullRLMPipeline", "PipelineResult", "IngestionPipeline"]

_LAZY: dict[str, tuple[str, str]] = {
    "FullRLMConfig": ("rlm.core.pipeline", "FullRLMConfig"),
    "FullRLMPipeline": ("rlm.core.pipeline", "FullRLMPipeline"),
    "PipelineResult": ("rlm.core.pipeline", "PipelineResult"),
    "IngestionPipeline": ("rlm.ingestion.pipeline", "IngestionPipeline"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
