"""Unit tests for the challenge regime→strategy map."""

from __future__ import annotations

import pytest

from rlm.challenge.challenge_strategy_map import (
    STRATEGY_MAP_CHALLENGE,
    get_challenge_strategy,
)

CANONICAL_DIRECTIONS = {"bull", "bear", "range", "transition"}
CANONICAL_VOL = {"high_vol", "low_vol", "transition"}
CANONICAL_LIQUIDITY = {"high_liquidity", "low_liquidity"}
CANONICAL_FLOW = {"supportive", "destabilizing"}


class TestCanonicalLabels:
    def test_all_map_keys_use_canonical_direction(self):
        for direction, *_ in STRATEGY_MAP_CHALLENGE:
            assert direction in CANONICAL_DIRECTIONS, f"Non-canonical direction: {direction!r}"

    def test_all_map_keys_use_canonical_vol(self):
        for _, vol, *_ in STRATEGY_MAP_CHALLENGE:
            assert vol in CANONICAL_VOL, f"Non-canonical vol: {vol!r}"

    def test_all_map_keys_use_canonical_liquidity(self):
        for _, _, liquidity, _ in STRATEGY_MAP_CHALLENGE:
            assert liquidity in CANONICAL_LIQUIDITY, f"Non-canonical liquidity: {liquidity!r}"

    def test_all_map_keys_use_canonical_flow(self):
        for _, _, _, flow in STRATEGY_MAP_CHALLENGE:
            assert flow in CANONICAL_FLOW, f"Non-canonical flow: {flow!r}"


class TestGetChallengeStrategy:
    @pytest.mark.parametrize("regime,expected", [
        (("bull", "low_vol",  "high_liquidity", "supportive"),    "aggressive_daytrader_call"),
        (("bull", "high_vol", "high_liquidity", "supportive"),    "aggressive_daytrader_0DTE_straddle"),
        (("bear", "low_vol",  "high_liquidity", "supportive"),    "aggressive_daytrader_put"),
        (("bear", "high_vol", "high_liquidity", "supportive"),    "aggressive_daytrader_put"),
        (("bear", "low_vol",  "high_liquidity", "destabilizing"), "aggressive_daytrader_put"),
        (("bear", "high_vol", "high_liquidity", "destabilizing"), "aggressive_daytrader_put"),
    ])
    def test_mapped_regimes(self, regime, expected):
        assert get_challenge_strategy(regime) == expected

    @pytest.mark.parametrize("regime", [
        ("range",      "low_vol",  "high_liquidity", "supportive"),
        ("transition", "high_vol", "high_liquidity", "supportive"),
        ("bull",       "high_vol", "low_liquidity",  "destabilizing"),
        ("bull",       "low_vol",  "high_liquidity", "destabilizing"),
    ])
    def test_unmapped_regimes_return_no_trade(self, regime):
        assert get_challenge_strategy(regime) == "no_trade"

    def test_all_mapped_values_are_known_strategies(self):
        known = {
            "aggressive_daytrader_call",
            "aggressive_daytrader_put",
            "aggressive_daytrader_0DTE_straddle",
        }
        for strategy in STRATEGY_MAP_CHALLENGE.values():
            assert strategy in known, f"Unknown strategy in map: {strategy!r}"
