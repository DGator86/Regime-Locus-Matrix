"""Unit tests for the four-stage persona pipeline."""

from __future__ import annotations

import pytest

from rlm.persona.models import PersonaPipelineInput
from rlm.persona.pipeline import PersonaConfig, PersonaDecisionPipeline


def _pipeline(cfg: PersonaConfig | None = None) -> PersonaDecisionPipeline:
    return PersonaDecisionPipeline(config=cfg)


def _bullish_inp(**overrides) -> PersonaPipelineInput:
    defaults = dict(
        symbol="SPY",
        regime_label="bull_trend",
        regime_confidence=0.82,
        forecast_return=0.015,
        realized_vol=0.18,
        signal_alignment=0.78,
        momentum_score=0.6,
        mean_reversion_score=0.0,
        dealer_gamma_exposure=0.3,
        options_put_call_ratio=0.9,
        bid_ask_spread_pct=0.01,
        volume_ratio=1.3,
        historical_edge=0.65,
    )
    defaults.update(overrides)
    return PersonaPipelineInput(**defaults)


def _bearish_inp(**overrides) -> PersonaPipelineInput:
    defaults = dict(
        symbol="QQQ",
        regime_label="bear_vol",
        regime_confidence=0.79,
        forecast_return=-0.012,
        realized_vol=0.25,
        signal_alignment=0.25,
        momentum_score=-0.5,
        mean_reversion_score=0.0,
        dealer_gamma_exposure=-0.3,
        options_put_call_ratio=1.6,
        bid_ask_spread_pct=0.012,
        volume_ratio=1.1,
        historical_edge=0.60,
    )
    defaults.update(overrides)
    return PersonaPipelineInput(**defaults)


class TestSevenStage:
    def test_bullish_aligned_gives_bullish_bias(self):
        result = _pipeline().run(_bullish_inp())
        assert result.seven.bias == "bullish"
        assert result.seven.confidence > 0.5

    def test_bearish_aligned_gives_bearish_bias(self):
        result = _pipeline().run(_bearish_inp())
        assert result.seven.bias == "bearish"

    def test_weak_signals_give_neutral(self):
        inp = _bullish_inp(
            regime_confidence=0.4,
            signal_alignment=0.5,
            momentum_score=0.0,
            forecast_return=0.0,
        )
        result = _pipeline().run(inp)
        # neutral or low confidence should push to no_trade via Sisko
        assert result.sisko.directive == "no_trade"


class TestGarakStage:
    def test_wide_spread_triggers_veto(self):
        inp = _bullish_inp(bid_ask_spread_pct=0.08)  # >5% → veto
        result = _pipeline().run(inp)
        assert result.garak.veto is True
        assert result.sisko.directive == "no_trade"

    def test_high_trap_risk_triggers_veto(self):
        # Very high PCR when bullish = trap signal
        inp = _bullish_inp(options_put_call_ratio=2.5, volume_ratio=0.5)
        result = _pipeline().run(inp)
        assert result.garak.trap_risk > 0.0

    def test_clean_setup_no_veto(self):
        result = _pipeline().run(_bullish_inp())
        assert result.garak.veto is False
        assert result.garak.trap_risk < 0.65


class TestSiskoStage:
    def test_bullish_aligned_gives_long(self):
        result = _pipeline().run(_bullish_inp())
        assert result.sisko.directive == "long"

    def test_bearish_aligned_gives_short(self):
        result = _pipeline().run(_bearish_inp())
        assert result.sisko.directive == "short"

    def test_low_confidence_gives_no_trade(self):
        inp = _bullish_inp(regime_confidence=0.25, signal_alignment=0.30)
        result = _pipeline().run(inp)
        assert result.sisko.directive == "no_trade"

    def test_veto_overrides_directive(self):
        inp = _bullish_inp(bid_ask_spread_pct=0.10)
        result = _pipeline().run(inp)
        assert result.sisko.directive == "no_trade"
        assert "veto" in result.sisko.reason.lower()


class TestDataStage:
    def test_high_edge_gives_high_regime_match(self):
        inp = _bullish_inp(historical_edge=0.72)
        result = _pipeline().run(inp)
        assert result.data.regime_match == "high"

    def test_low_edge_sets_review_flag_if_trading(self):
        # Force a tradeable setup but with low edge
        inp = _bullish_inp(historical_edge=0.25)
        result = _pipeline().run(inp)
        if result.sisko.directive != "no_trade":
            assert result.data.review_flag is True

    def test_elevated_vol_gives_size_down_note(self):
        inp = _bullish_inp(realized_vol=0.45)
        result = _pipeline().run(inp)
        assert "size" in result.data.adaptation_note.lower() or "vol" in result.data.adaptation_note.lower()

    def test_serialisation(self):
        result = _pipeline().run(_bullish_inp())
        d = result.to_dict()
        assert "seven" in d
        assert "garak" in d
        assert "sisko" in d
        assert "data" in d
        assert d["symbol"] == "SPY"
