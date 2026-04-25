"""
ChallengeDecisionPipeline — orchestrates setup ranking, PDT gating,
trade-mode selection, contract profiling, and risk plan generation.
"""

from __future__ import annotations

from rlm.challenge.models import (
    ChallengeAccountState,
    ChallengePipelineConfig,
    ChallengeDirective,
    ContractProfileRecommendation,
    PDTTracker,
    RiskPlan,
    SetupScoreResult,
    StageSizingRule,
    TradeModeDecision,
)
from rlm.persona.models import PersonaPipelineResult


class ChallengeDecisionPipeline:
    """Full challenge decision stack.

    Usage::

        from rlm.challenge import ChallengeDecisionPipeline, ChallengePipelineConfig
        from rlm.challenge.state import ChallengeStateManager

        mgr = ChallengeStateManager()
        state, pdt = mgr.load()
        pipeline = ChallengeDecisionPipeline()
        directive = pipeline.run("SPY", persona_result, state, pdt)
    """

    def __init__(self, config: ChallengePipelineConfig | None = None) -> None:
        self._cfg = config or ChallengePipelineConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        persona: PersonaPipelineResult,
        state: ChallengeAccountState,
        pdt: PDTTracker,
    ) -> ChallengeDirective:
        """Evaluate a single symbol and return a ChallengeDirective."""
        cfg = self._cfg

        # 0. Universe gate
        if symbol.upper() not in [u.upper() for u in cfg.allowed_universe]:
            return self._no_trade(symbol, pdt, "symbol not in challenge universe")

        # 1. Persona veto passthrough
        if persona.sisko.directive == "no_trade":
            return self._no_trade(symbol, pdt, f"persona no_trade: {persona.sisko.entry_policy}")

        # 2. Setup scoring
        score_result = self._score_setup(persona)
        if not score_result.passed_threshold:
            return self._no_trade(symbol, pdt, f"setup score {score_result.setup_score:.2f} below min")

        # 3. Trade mode (scalp vs swing)
        mode_decision = self._decide_mode(score_result, pdt)

        # 4. Contract profile
        contract_profile = self._contract_profile(mode_decision)

        # 5. Risk plan from current account stage
        sizing_rule = self._sizing_rule(state)
        risk_plan = RiskPlan(
            premium_outlay_pct=sizing_rule.premium_outlay_pct,
            max_account_loss_pct=sizing_rule.max_loss_pct,
            hard_stop_pct=cfg.hard_stop_pct,
            trail_activate_pct=cfg.trail_activate_pct,
            trail_drawdown_pct=cfg.trail_drawdown_pct,
            profit_target_pct=cfg.profit_target_pct,
            partial_take_pct=cfg.partial_take_pct,
            use_underlying_invalidation=True,
            force_close_dte_threshold=1,
        )

        directive_val = persona.sisko.directive  # "long" or "short"
        reason = (
            f"score={score_result.setup_score:.2f} conviction={score_result.conviction} "
            f"mode={mode_decision.trade_mode} pdt_remain={pdt.day_trades_remaining}"
        )

        return ChallengeDirective(
            symbol=symbol.upper(),
            setup_score=round(score_result.setup_score, 4),
            conviction=score_result.conviction,
            directive=directive_val,  # type: ignore[arg-type]
            trade_mode=mode_decision.trade_mode,
            same_day_exit_allowed=mode_decision.same_day_exit_allowed,
            pdt_slots_remaining=pdt.day_trades_remaining,
            contract_profile=contract_profile,
            risk_plan=risk_plan,
            reason_summary=reason,
        )

    # ------------------------------------------------------------------
    # Setup scoring
    # ------------------------------------------------------------------

    def _score_setup(self, persona: PersonaPipelineResult) -> SetupScoreResult:
        cfg = self._cfg
        s = persona.seven
        g = persona.garak
        d = persona.data

        # Dealer support component: 1.0 if supportive, 0.5 neutral, 0.0 opposed
        dealer_map = {"supportive": 1.0, "neutral": 0.5, "opposed": 0.0}
        dealer_support = dealer_map.get(g.dealer_alignment, 0.5)

        raw_score = (
            cfg.weight_seven_confidence * s.confidence
            + cfg.weight_signal_alignment * s.signal_alignment
            + cfg.weight_historical_edge * d.historical_edge
            + cfg.weight_dealer_support * dealer_support
            - 0.30 * g.trap_risk          # trap penalty
        )
        score = max(0.0, min(1.0, raw_score))

        if score >= cfg.elite_setup_score:
            conviction = "elite"
        elif score >= 0.65:
            conviction = "high"
        elif score >= cfg.min_setup_score:
            conviction = "medium"
        else:
            conviction = "low"

        return SetupScoreResult(
            setup_score=round(score, 4),
            conviction=conviction,
            passed_threshold=score >= cfg.min_setup_score,
            ranking_reason=(
                f"conf={s.confidence:.2f} align={s.signal_alignment:.2f} "
                f"edge={d.historical_edge:.2f} dealer={g.dealer_alignment} "
                f"trap={g.trap_risk:.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # Trade mode
    # ------------------------------------------------------------------

    def _decide_mode(self, score: SetupScoreResult, pdt: PDTTracker) -> TradeModeDecision:
        cfg = self._cfg

        if score.conviction == "elite" and pdt.same_day_exit_allowed:
            return TradeModeDecision(
                trade_mode="scalp",
                same_day_exit_allowed=True,
                pdt_reason=f"elite conviction + {pdt.day_trades_remaining} PDT slots",
            )
        elif pdt.same_day_exit_allowed:
            # Strong but not elite → swing; can still take same-day profit
            return TradeModeDecision(
                trade_mode="swing_candidate",
                same_day_exit_allowed=True,
                pdt_reason="swing — same-day profit allowed if target hit",
            )
        else:
            return TradeModeDecision(
                trade_mode="swing_candidate",
                same_day_exit_allowed=False,
                pdt_reason="no PDT slots — must be willing to hold overnight",
            )

    # ------------------------------------------------------------------
    # Contract profile
    # ------------------------------------------------------------------

    def _contract_profile(self, mode: TradeModeDecision) -> ContractProfileRecommendation:
        cfg = self._cfg
        if mode.trade_mode == "scalp":
            return ContractProfileRecommendation(
                target_delta_min=cfg.scalp_delta_min,
                target_delta_max=cfg.scalp_delta_max,
                preferred_dte_min=cfg.scalp_dte_min,
                preferred_dte_max=cfg.scalp_dte_max,
                max_spread_pct=cfg.max_spread_pct,
                liquidity_tier="high",
                note="scalp mode — tighter delta, shorter DTE, must have tight spread",
            )
        return ContractProfileRecommendation(
            target_delta_min=cfg.swing_delta_min,
            target_delta_max=cfg.swing_delta_max,
            preferred_dte_min=cfg.swing_dte_min,
            preferred_dte_max=cfg.swing_dte_max,
            max_spread_pct=cfg.max_spread_pct,
            liquidity_tier="high",
            note="swing mode — higher delta, more DTE buffer for overnight holds",
        )

    # ------------------------------------------------------------------
    # Sizing tier
    # ------------------------------------------------------------------

    def _sizing_rule(self, state: ChallengeAccountState) -> StageSizingRule:
        for rule in self._cfg.stage_sizing:
            if rule.equity_min <= state.current_equity < rule.equity_max:
                return rule
        # Fallback: last rule
        return self._cfg.stage_sizing[-1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _no_trade(self, symbol: str, pdt: PDTTracker, reason: str) -> ChallengeDirective:
        cfg = self._cfg
        empty_profile = ContractProfileRecommendation(
            target_delta_min=0.0, target_delta_max=0.0,
            preferred_dte_min=0, preferred_dte_max=0,
            max_spread_pct=0.0, liquidity_tier="low", note="n/a",
        )
        empty_risk = RiskPlan(
            premium_outlay_pct=0.0, max_account_loss_pct=0.0,
            hard_stop_pct=cfg.hard_stop_pct, trail_activate_pct=cfg.trail_activate_pct,
            trail_drawdown_pct=cfg.trail_drawdown_pct, profit_target_pct=cfg.profit_target_pct,
            partial_take_pct=cfg.partial_take_pct, use_underlying_invalidation=False,
            force_close_dte_threshold=1,
        )
        return ChallengeDirective(
            symbol=symbol.upper(),
            setup_score=0.0,
            conviction="low",
            directive="no_trade",
            trade_mode="no_trade",
            same_day_exit_allowed=pdt.same_day_exit_allowed,
            pdt_slots_remaining=pdt.day_trades_remaining,
            contract_profile=empty_profile,
            risk_plan=empty_risk,
            reason_summary=reason,
        )
