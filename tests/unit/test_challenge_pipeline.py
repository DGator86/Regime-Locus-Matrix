"""Unit tests for the challenge decision pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rlm.challenge.models import ChallengeAccountState, PDTTracker
from rlm.challenge.pipeline import ChallengeDecisionPipeline
from rlm.persona.models import (
    DataStageOutput,
    GarakStageOutput,
    PersonaPipelineResult,
    SevenStageOutput,
    SiskoStageOutput,
)


def _make_persona(
    *,
    bias: str = "bullish",
    signal_alignment: float = 0.78,
    confidence: float = 0.82,
    trap_risk: float = 0.10,
    dealer_alignment: str = "supportive",
    veto: bool = False,
    directive: str = "long",
    historical_edge: float = 0.65,
) -> PersonaPipelineResult:
    """Construct a PersonaPipelineResult directly — no live pipeline needed."""
    return PersonaPipelineResult(
        seven=SevenStageOutput(bias=bias, signal_alignment=signal_alignment, confidence=confidence),  # type: ignore[arg-type]
        garak=GarakStageOutput(
            trap_risk=trap_risk,
            dealer_alignment=dealer_alignment,  # type: ignore[arg-type]
            liquidity_comment="normal",
            veto=veto,
        ),
        sisko=SiskoStageOutput(
            directive=directive,  # type: ignore[arg-type]
            entry_policy="enter on confirmation",
            invalidation_policy="close on reversal",
            target_policy="2x premium",
        ),
        data=DataStageOutput(
            regime_match="high",
            historical_edge=historical_edge,
            adaptation_note="",
            review_flag=False,
        ),
    )


def _run(
    symbol: str = "SPY",
    pdt_slots: int = 3,
    equity: float = 1_000.0,
    **persona_overrides,
) -> object:
    """Helper: build a challenge directive for given symbol / persona inputs."""
    persona = _make_persona(
        bias=persona_overrides.pop("bias", "bullish"),
        signal_alignment=persona_overrides.pop("signal_alignment", 0.78),
        confidence=persona_overrides.pop("regime_confidence", 0.82),
        trap_risk=persona_overrides.pop("trap_risk", 0.10),
        dealer_alignment=persona_overrides.pop("dealer_alignment", "supportive"),
        veto=persona_overrides.pop("veto", False),
        directive=persona_overrides.pop("directive", "long"),
        historical_edge=persona_overrides.pop("historical_edge", 0.65),
    )
    state = ChallengeAccountState(current_equity=equity)
    used_slots = max(0, 3 - pdt_slots)
    pdt = PDTTracker(day_trades_used_last_5d=[used_slots])
    return ChallengeDecisionPipeline().run(symbol, persona, state, pdt)


class TestChallengeUniverse:
    def test_unsupported_symbol_is_no_trade(self):
        d = _run(symbol="GME")  # not in default universe
        assert d.directive == "no_trade"
        assert "universe" in d.reason_summary.lower()

    def test_supported_symbol_passes(self):
        d = _run(symbol="SPY")
        assert d.symbol == "SPY"


class TestSetupScoring:
    def test_elite_bullish_gives_scalp_with_pdt(self):
        d = _run(
            symbol="SPY",
            pdt_slots=3,
            regime_confidence=0.90,
            signal_alignment=0.88,
            historical_edge=0.72,
        )
        assert d.directive == "long"
        assert d.trade_mode == "scalp"
        assert d.conviction == "elite"

    def test_strong_bullish_no_pdt_gives_swing(self):
        d = _run(
            symbol="SPY",
            pdt_slots=0,  # PDT exhausted
            regime_confidence=0.82,
            signal_alignment=0.78,
        )
        assert d.directive == "long"
        assert d.trade_mode == "swing_candidate"
        assert d.same_day_exit_allowed is False

    def test_weak_confidence_gives_no_trade(self):
        d = _run(
            symbol="SPY",
            regime_confidence=0.28,
            signal_alignment=0.30,
            historical_edge=0.20,
        )
        assert d.directive == "no_trade"

    def test_garak_veto_gives_no_trade(self):
        d = _run(
            symbol="QQQ",
            veto=True,  # Garak hard veto
            directive="no_trade",
        )
        assert d.directive == "no_trade"


class TestBearishCase:
    def test_bearish_elite_gives_short(self):
        persona = _make_persona(
            bias="bearish",
            signal_alignment=0.80,
            confidence=0.85,
            trap_risk=0.08,
            dealer_alignment="supportive",
            directive="short",
            historical_edge=0.64,
        )
        state = ChallengeAccountState(current_equity=5_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        d = ChallengeDecisionPipeline().run("SPY", persona, state, pdt)
        assert d.directive == "short"


class TestPDTTracker:
    def test_pdt_tracker_blocks_same_day_when_exhausted(self):
        pdt = PDTTracker(day_trades_used_last_5d=[3])
        assert pdt.day_trades_remaining == 0
        assert pdt.same_day_exit_allowed is False
        assert pdt.must_hold_overnight_if_entered is True

    def test_pdt_tracker_allows_when_slots_remain(self):
        pdt = PDTTracker(day_trades_used_last_5d=[1])
        assert pdt.day_trades_remaining == 2
        assert pdt.same_day_exit_allowed is True

    def test_new_session_rolls_window(self):
        pdt = PDTTracker(day_trades_used_last_5d=[1, 1, 1, 1, 1])
        pdt.new_session()
        assert len(pdt.day_trades_used_last_5d) == 5
        assert pdt.day_trades_used_last_5d[-1] == 0


class TestContractProfile:
    def test_scalp_mode_has_tighter_delta_and_short_dte(self):
        # Elite setup to force scalp (elite_setup_score now 0.70)
        d = _run(
            symbol="SPY",
            pdt_slots=3,
            regime_confidence=0.92,
            signal_alignment=0.90,
            historical_edge=0.75,
        )
        if d.trade_mode == "scalp":
            assert d.contract_profile.target_delta_max <= 0.65
            assert d.contract_profile.preferred_dte_max <= 3  # scalp_dte_max=3 now

    def test_swing_mode_has_wider_delta(self):
        d = _run(symbol="SPY", pdt_slots=0)  # no PDT → swing
        if d.trade_mode == "swing_candidate":
            assert d.contract_profile.target_delta_min >= 0.40


class TestStageSizing:
    def test_stage1_sizing_applied(self):
        d = _run(
            symbol="SPY",
            equity=1_200.0,
            pdt_slots=3,
            regime_confidence=0.90,
            signal_alignment=0.88,
            historical_edge=0.72,
        )
        if d.directive != "no_trade":
            assert d.risk_plan.premium_outlay_pct == pytest.approx(0.12)

    def test_stage2_sizing_applied(self):
        d = _run(
            symbol="SPY",
            equity=4_500.0,
            pdt_slots=3,
            regime_confidence=0.90,
            signal_alignment=0.88,
            historical_edge=0.72,
        )
        if d.directive != "no_trade":
            assert d.risk_plan.premium_outlay_pct == pytest.approx(0.15)

    def test_stage3_sizing_applied(self):
        d = _run(
            symbol="SPY",
            equity=12_000.0,
            pdt_slots=3,
            regime_confidence=0.90,
            signal_alignment=0.88,
            historical_edge=0.72,
        )
        if d.directive != "no_trade":
            assert d.risk_plan.premium_outlay_pct == pytest.approx(0.18)


class TestSniperGate:
    """Backward-compatibility and sniper gate behaviour tests."""

    def test_no_intraday_kwargs_bypasses_gate(self):
        """Existing callers without intraday kwargs are unaffected."""
        d = _run(symbol="SPY")
        assert d.directive in ("long", "short", "no_trade")

    def test_sniper_filter_fail_forces_no_trade(self):
        """When is_great_daytrade_setup returns False the directive is no_trade."""
        persona = _make_persona()
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        regime: tuple[str, str, str, str] = ("bull", "low_vol", "high_liquidity", "supportive")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=False):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=regime,
            )
        assert d.directive == "no_trade"
        assert "sniper" in d.reason_summary.lower()

    def test_unmapped_regime_forces_no_trade(self):
        """A regime absent from STRATEGY_MAP_CHALLENGE returns no_trade even if filter passes."""
        persona = _make_persona()
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        unmapped: tuple[str, str, str, str] = ("range", "low_vol", "high_liquidity", "supportive")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=True):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=unmapped,
            )
        assert d.directive == "no_trade"

    def test_sniper_strategy_tagged_in_reason(self):
        """When gate passes and regime maps to a strategy, its name appears in reason_summary."""
        persona = _make_persona(directive="long", signal_alignment=0.78, confidence=0.82)
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        regime: tuple[str, str, str, str] = ("bull", "low_vol", "high_liquidity", "supportive")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=True):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=regime,
            )
        assert "aggressive_daytrader_call" in d.reason_summary

    def test_sniper_conflicting_persona_directive_forces_no_trade(self):
        """A regime-mapped call must not execute through a short persona directive."""
        persona = _make_persona(directive="short", bias="bearish", signal_alignment=0.78, confidence=0.82)
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        regime: tuple[str, str, str, str] = ("bull", "low_vol", "high_liquidity", "supportive")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=True):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=regime,
            )
        assert d.directive == "no_trade"
        assert "conflicts" in d.reason_summary

    def test_sniper_straddle_strategy_forces_no_trade_until_directive_support_exists(self):
        """ChallengeDirective cannot currently express multi-leg straddles safely."""
        persona = _make_persona(directive="long", signal_alignment=0.78, confidence=0.82)
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        regime: tuple[str, str, str, str] = ("bull", "high_vol", "high_liquidity", "supportive")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=True):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=regime,
            )
        assert d.directive == "no_trade"
        assert "not supported" in d.reason_summary

    def test_bearish_destabilizing_regime_passes_sniper(self):
        """Bear + destabilizing dealer flow is mapped and resolves to a put strategy."""
        persona = _make_persona(directive="short", bias="bearish")
        state = ChallengeAccountState(current_equity=1_000.0)
        pdt = PDTTracker(day_trades_used_last_5d=[0])
        regime: tuple[str, str, str, str] = ("bear", "high_vol", "high_liquidity", "destabilizing")

        with patch("rlm.challenge.pipeline.is_great_daytrade_setup", return_value=True):
            d = ChallengeDecisionPipeline().run(
                "SPY", persona, state, pdt,
                current_bar=object(), intraday_df=object(), regime=regime,
            )
        assert "aggressive_daytrader_put" in d.reason_summary
