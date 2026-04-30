"""Optimization entry points for tuning and nightly overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlm.optimization.config import NightlyHyperparams

if TYPE_CHECKING:
    from rlm.optimization.nightly import NightlyMTFOptimizer

__all__ = [
    "NightlyHyperparams",
    "NightlyMTFOptimizer",
]


def __getattr__(name: str) -> Any:
    if name == "NightlyMTFOptimizer":
        from rlm.optimization.nightly import NightlyMTFOptimizer

        return NightlyMTFOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
