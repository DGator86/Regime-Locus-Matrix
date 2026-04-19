from __future__ import annotations

from rlm.roee.regime_persistence import (
    RegimePersistenceState,
    persistence_size_multiplier,
    persistence_trade_gate,
)


def test_persistence_gate_blocks_unstable_regime() -> None:
    state = RegimePersistenceState(
        regime="transition",
        consecutive_bars=1,
        dominant_probability=0.35,
        flip_rate_recent=0.8,
    )
    assert persistence_trade_gate(state) is False


def test_persistence_multiplier_rewards_stable_regime() -> None:
    state = RegimePersistenceState(
        regime="trend_up_stable",
        consecutive_bars=6,
        dominant_probability=0.7,
        flip_rate_recent=0.1,
    )
    assert persistence_size_multiplier(state) > 1.0
