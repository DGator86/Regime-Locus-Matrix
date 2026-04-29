"""Unit tests for the four-stage persona decision pipeline."""

from __future__ import annotations

from rlm.persona.config import PersonaConfig
from rlm.persona.models import PersonaInputs
from rlm.persona.pipeline import PersonaDecisionPipeline
from rlm.persona.stages import run_data, run_garak, run_seven, run_sisko

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bullish_inputs(**overrides: object) -> PersonaInputs:
    base = dict(
        s_d=0.75,
        s_v=-0.40,
        s_l=0.65,
        s_g=0.55,
        direction_regime="bull",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="supportive",
        hmm_confidence=0.82,
        roee_action="enter",
        backtest_metrics=None,
    )
    base.update(overrides)
    return PersonaInputs(**base)  # type: ignore[arg-type]


def _bearish_inputs(**overrides: object) -> PersonaInputs:
    base = dict(
        s_d=-0.70,
        s_v=-0.20,
        s_l=0.50,
        s_g=-0.45,
        direction_regime="bear",
        volatility_regime="low_vol",
        liquidity_regime="high_liquidity",
        dealer_flow_regime="opposed",
        hmm_confidence=0.78,
        roee_action="enter",
        backtest_metrics=None,
    )
    base.update(overrides)
    return PersonaInputs(**base)  # type: ignore[arg-type]


CFG = PersonaConfig()


# ---------------------------------------------------------------------------
# Seven stage tests
# ---------------------------------------------------------------------------


class TestSevenStage:
    def test_bullish_bias_from_regime(self) -> None:
        out = run_seven(_bullish_inputs(), CFG)
        assert out.bias == "bullish"

    def test_bearish_bias_from_regime(self) -> None:
        out = run_seven(_bearish_inputs(), CFG)
        assert out.bias == "bearish"

    def test_neutral_bias_when_scores_flat(self) -> None:
        inp = _bullish_inputs(s_d=0.05, direction_regime="neutral")
        out = run_seven(inp, CFG)
        assert out.bias == "neutral"

    def test_signal_alignment_high_for_clean_bull(self) -> None:
        out = run_seven(_bullish_inputs(), CFG)
        assert out.signal_alignment >= CFG.signal_alignment_threshold

    def test_signal_alignment_zero_for_neutral(self) -> None:
        inp = _bullish_inputs(s_d=0.0, direction_regime="neutral")
        out = run_seven(inp, CFG)
        assert out.signal_alignment == 0.0

    def test_confidence_bounded_zero_to_one(self) -> None:
        for s_d in (-1.0, 0.0, 0.5, 1.0):
            for hmm in (0.0, 0.5, 1.0):
                inp = _bullish_inputs(s_d=s_d, hmm_confidence=hmm)
                out = run_seven(inp, CFG)
                assert 0.0 <= out.confidence <= 1.0

    def test_s_d_overrides_neutral_regime_bullish(self) -> None:
        inp = _bullish_inputs(direction_regime="neutral", s_d=0.8)
        out = run_seven(inp, CFG)
        assert out.bias == "bullish"

    def test_s_d_overrides_neutral_regime_bearish(self) -> None:
        inp = _bullish_inputs(direction_regime="neutral", s_d=-0.8)
        out = run_seven(inp, CFG)
        assert out.bias == "bearish"


# ---------------------------------------------------------------------------
# Garak stage tests
# ---------------------------------------------------------------------------


class TestGarakStage:
    def test_low_trap_risk_for_clean_bull(self) -> None:
        inp = _bullish_inputs()
        seven = run_seven(inp, CFG)
        out = run_garak(inp, seven, CFG)
        assert out.trap_risk < CFG.trap_risk_veto_threshold
        assert not out.veto

    def test_high_trap_risk_triggers_veto(self) -> None:
        inp = _bullish_inputs(
            s_v=0.90,  # high vol stress
            s_l=-0.90,  # poor liquidity
            dealer_flow_regime="opposed",
        )
        seven = run_seven(inp, CFG)
        out = run_garak(inp, seven, CFG)
        assert out.trap_risk >= CFG.trap_risk_veto_threshold
        assert out.veto

    def test_dealer_alignment_label_supportive(self) -> None:
        inp = _bullish_inputs(dealer_flow_regime="supportive")
        seven = run_seven(inp, CFG)
        out = run_garak(inp, seven, CFG)
        assert out.dealer_alignment == "supportive"

    def test_dealer_alignment_label_opposed(self) -> None:
        inp = _bullish_inputs(dealer_flow_regime="opposed")
        seven = run_seven(inp, CFG)
        out = run_garak(inp, seven, CFG)
        assert out.dealer_alignment == "opposed"

    def test_trap_risk_bounded(self) -> None:
        for s_v in (-1.0, 0.0, 1.0):
            for s_l in (-1.0, 0.0, 1.0):
                inp = _bullish_inputs(s_v=s_v, s_l=s_l)
                seven = run_seven(inp, CFG)
                out = run_garak(inp, seven, CFG)
                assert 0.0 <= out.trap_risk <= 1.0

    def test_liquidity_comment_reflects_s_l(self) -> None:
        good = _bullish_inputs(s_l=0.8)
        bad = _bullish_inputs(s_l=-0.8)
        seven = run_seven(good, CFG)
        assert "clean" in run_garak(good, seven, CFG).liquidity_comment
        seven_bad = run_seven(bad, CFG)
        assert "stressed" in run_garak(bad, seven_bad, CFG).liquidity_comment


# ---------------------------------------------------------------------------
# Sisko stage tests
# ---------------------------------------------------------------------------


class TestSiskoStage:
    def test_bullish_aligned_produces_long_directive(self) -> None:
        inp = _bullish_inputs()
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "long"

    def test_bearish_aligned_produces_short_directive(self) -> None:
        # Negative S_L (thin liquidity) and large |S_D|/|S_G| make all three
        # scores bearish-aligned so signal_alignment clears the threshold.
        inp = _bearish_inputs(
            dealer_flow_regime="neutral",
            s_d=-0.90,
            s_v=-0.30,
            s_l=-0.50,
            s_g=-0.70,
        )
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "short"

    def test_garak_veto_forces_no_trade(self) -> None:
        inp = _bullish_inputs(s_v=0.95, s_l=-0.95, dealer_flow_regime="opposed")
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        assert garak.veto
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "no_trade"

    def test_weak_confidence_produces_no_trade(self) -> None:
        inp = _bullish_inputs(hmm_confidence=0.05, s_d=0.10)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "no_trade"

    def test_weak_alignment_produces_no_trade(self) -> None:
        # Bullish regime but all supporting scores are near zero
        inp = _bullish_inputs(s_l=0.0, s_g=0.0, s_d=0.05)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "no_trade"

    def test_no_trade_policies_are_sensible(self) -> None:
        inp = _bullish_inputs(hmm_confidence=0.0, s_d=0.0)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert "stand aside" in out.entry_policy
        assert out.invalidation_policy == "n/a"

    def test_neutral_bias_produces_no_trade(self) -> None:
        inp = _bullish_inputs(s_d=0.05, direction_regime="neutral")
        seven = run_seven(inp, CFG)
        assert seven.bias == "neutral"
        garak = run_garak(inp, seven, CFG)
        out = run_sisko(inp, seven, garak, CFG)
        assert out.directive == "no_trade"


# ---------------------------------------------------------------------------
# Data stage tests
# ---------------------------------------------------------------------------


class TestDataStage:
    def test_high_backtest_metrics_produce_high_match(self) -> None:
        inp = _bullish_inputs(backtest_metrics={"sharpe_ratio": 1.8, "win_rate": 0.72})
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        assert out.regime_match == "high"
        assert out.historical_edge >= CFG.regime_match_high_threshold

    def test_poor_backtest_metrics_flag_review(self) -> None:
        inp = _bullish_inputs(backtest_metrics={"sharpe_ratio": 0.1, "win_rate": 0.35})
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        assert out.review_flag is True

    def test_no_backtest_metrics_falls_back_gracefully(self) -> None:
        inp = _bullish_inputs(backtest_metrics=None)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        assert 0.0 <= out.historical_edge <= 1.0
        assert out.adaptation_note  # non-empty

    def test_high_vol_regime_note(self) -> None:
        inp = _bullish_inputs(volatility_regime="high_vol", backtest_metrics=None)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        assert "elevated vol" in out.adaptation_note

    def test_low_vol_regime_note(self) -> None:
        inp = _bullish_inputs(volatility_regime="low_vol", backtest_metrics=None)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        assert "expanding vol" in out.adaptation_note

    def test_regime_match_low_for_weak_signals(self) -> None:
        inp = _bullish_inputs(s_d=0.05, s_l=0.05, s_g=0.05, backtest_metrics=None)
        seven = run_seven(inp, CFG)
        garak = run_garak(inp, seven, CFG)
        sisko = run_sisko(inp, seven, garak, CFG)
        out = run_data(inp, sisko, CFG)
        # Weak signals → low historical edge → moderate or low
        assert out.regime_match in ("low", "moderate")


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestPersonaDecisionPipeline:
    def test_bullish_scenario_end_to_end(self) -> None:
        inp = _bullish_inputs()
        result = PersonaDecisionPipeline().run_from_inputs(inp)
        assert result.sisko.directive == "long"
        assert result.seven.bias == "bullish"
        assert not result.garak.veto

    def test_bearish_scenario_end_to_end(self) -> None:
        inp = _bearish_inputs(
            dealer_flow_regime="neutral",
            s_d=-0.90,
            s_v=-0.30,
            s_l=-0.50,
            s_g=-0.70,
        )
        result = PersonaDecisionPipeline().run_from_inputs(inp)
        assert result.sisko.directive == "short"
        assert result.seven.bias == "bearish"

    def test_high_trap_no_trade(self) -> None:
        inp = _bullish_inputs(s_v=0.95, s_l=-0.95, dealer_flow_regime="opposed")
        result = PersonaDecisionPipeline().run_from_inputs(inp)
        assert result.sisko.directive == "no_trade"
        assert result.garak.veto

    def test_weak_confidence_no_trade(self) -> None:
        inp = _bullish_inputs(hmm_confidence=0.05, s_d=0.08)
        result = PersonaDecisionPipeline().run_from_inputs(inp)
        assert result.sisko.directive == "no_trade"

    def test_to_dict_structure(self) -> None:
        inp = _bullish_inputs()
        result = PersonaDecisionPipeline().run_from_inputs(inp)
        d = result.to_dict()
        assert set(d) == {"seven", "garak", "sisko", "data"}
        assert "bias" in d["seven"]
        assert "signal_alignment" in d["seven"]
        assert "confidence" in d["seven"]
        assert "trap_risk" in d["garak"]
        assert "veto" in d["garak"]
        assert "directive" in d["sisko"]
        assert "historical_edge" in d["data"]
        assert "review_flag" in d["data"]

    def test_custom_config_lower_veto_threshold(self) -> None:
        strict_cfg = PersonaConfig(trap_risk_veto_threshold=0.30)
        # Moderate stress that would not veto with default config
        inp = _bullish_inputs(s_v=0.30, s_l=-0.20)
        default_result = PersonaDecisionPipeline().run_from_inputs(inp)
        strict_result = PersonaDecisionPipeline(strict_cfg).run_from_inputs(inp)
        # Strict config should be at least as restrictive
        if default_result.sisko.directive == "long":
            # strict might produce no_trade due to lower veto threshold
            assert strict_result.sisko.directive in ("long", "no_trade")

    def test_run_from_pipeline_result_empty_dfs(self) -> None:
        """Pipeline degrades gracefully when DataFrames are empty."""
        import pandas as pd

        class FakePipelineResult:
            factors_df = pd.DataFrame()
            forecast_df = pd.DataFrame()
            policy_df = pd.DataFrame()
            backtest_metrics = None

        result = PersonaDecisionPipeline().run(FakePipelineResult())
        assert result.sisko.directive == "no_trade"
        assert result.seven.bias == "neutral"
