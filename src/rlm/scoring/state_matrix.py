"""Backward-compatibility re-export. Canonical location: rlm.features.scoring.state_matrix."""

from rlm.features.scoring.state_matrix import classify_state_matrix, make_regime_key, regime_is_tradeable

__all__ = ["classify_state_matrix", "make_regime_key", "regime_is_tradeable"]
