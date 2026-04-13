"""Backward-compatibility re-export. Canonical location: rlm.data.bars_enrichment."""

from rlm.data.bars_enrichment import (
    enrich_bars_from_option_chain,
    enrich_bars_with_surface_features,
    enrich_bars_with_vix,
    prepare_bars_for_factors,
)

__all__ = [
    "enrich_bars_from_option_chain",
    "enrich_bars_with_surface_features",
    "enrich_bars_with_vix",
    "prepare_bars_for_factors",
]
