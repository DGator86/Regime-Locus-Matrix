"""
Deterministic four-stage persona decision pipeline.

Seven → Garak → Sisko → Data
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm.persona.models import (
    DataStageOutput,
    GarakStageOutput,
    PersonaPipelineInput,
    PersonaPipelineResult,
    SiskoStageOutput,
    SevenStageOutput,
)


@dataclass(frozen=True)
class PersonaConfig:
    """Configurable thresholds for the persona pipeline."""
    # Seven
    bullish_threshold: float = 0.55      # signal_alignment + confidence weighted
    bearish_threshold: float = 0.45
    confidence_floor: float = 0.40       # below this → no_trade

    # Garak
    trap_risk_veto_threshold: float = 0.65   # trap_risk above this → veto
    spread_veto_threshold: float = 0.05      # bid_ask > 5% → flag as illiquid
    put_call_veto_above: float = 1.8         # PCR above this → extreme fear

    # Sisko
    min_confidence_to_trade: float = 0.45

    # Data
    high_edge_threshold: float = 0.60
    low_edge_threshold: float = 0.40


class _SevenStage:
    """Normalise and interpret existing RLM signal outputs."""

    def __init__(self, cfg: PersonaConfig) -> None:
        self._cfg = cfg

    def run(self, inp: PersonaPipelineInput) -> SevenStageOutput:
        cfg = self._cfg

        # Combined confidence: regime confidence + signal alignment weighted
        confidence = 0.6 * inp.regime_confidence + 0.4 * inp.signal_alignment
        alignment = inp.signal_alignment

        # Determine directional bias
        bullish_score = (
            0.5 * max(inp.forecast_return, 0.0)
            + 0.3 * max(inp.momentum_score, 0.0)
            - 0.2 * max(-inp.mean_reversion_score, 0.0)
        )
        bearish_score = (
            0.5 * abs(min(inp.forecast_return, 0.0))
            + 0.3 * abs(min(inp.momentum_score, 0.0))
            - 0.2 * max(inp.mean_reversion_score, 0.0)
        )

        if bullish_score > bearish_score and alignment >= cfg.bullish_threshold:
            bias = "bullish"
        elif bearish_score > bullish_score and (1.0 - alignment) >= (1.0 - cfg.bearish_threshold):
            bias = "bearish"
        else:
            bias = "neutral"

        notes = (
            f"forecast_return={inp.forecast_return:.3f} "
            f"momentum={inp.momentum_score:.3f} "
            f"vol={inp.realized_vol:.3f}"
        )
        return SevenStageOutput(
            bias=bias,
            signal_alignment=round(alignment, 4),
            confidence=round(confidence, 4),
            regime_label=inp.regime_label,
            notes=notes,
        )


class _GarakStage:
    """Detect traps, false breakouts, and adversarial dealer behaviour."""

    def __init__(self, cfg: PersonaConfig) -> None:
        self._cfg = cfg

    def run(self, inp: PersonaPipelineInput, seven: SevenStageOutput) -> GarakStageOutput:
        cfg = self._cfg

        # --- Trap risk score ---
        trap_components: list[float] = []

        # Spread wideness signal
        spread_signal = min(inp.bid_ask_spread_pct / 0.05, 1.0)
        trap_components.append(spread_signal * 0.30)

        # PCR extremity: high PCR when bullish = potential bear trap
        if seven.bias == "bullish" and inp.options_put_call_ratio > 1.2:
            pcr_signal = min((inp.options_put_call_ratio - 1.2) / 0.6, 1.0)
            trap_components.append(pcr_signal * 0.35)
        elif seven.bias == "bearish" and inp.options_put_call_ratio < 0.8:
            pcr_signal = min((0.8 - inp.options_put_call_ratio) / 0.3, 1.0)
            trap_components.append(pcr_signal * 0.35)
        else:
            trap_components.append(0.0)

        # Low volume ratio = low conviction
        if inp.volume_ratio < 0.8:
            trap_components.append(0.2 * (1.0 - inp.volume_ratio))
        else:
            trap_components.append(0.0)

        # Misaligned dealer gamma
        if seven.bias == "bullish" and inp.dealer_gamma_exposure < -0.2:
            trap_components.append(0.15)
        elif seven.bias == "bearish" and inp.dealer_gamma_exposure > 0.2:
            trap_components.append(0.15)
        else:
            trap_components.append(0.0)

        trap_risk = min(sum(trap_components), 1.0)
        veto = trap_risk >= cfg.trap_risk_veto_threshold

        # Dealer alignment
        if inp.dealer_gamma_exposure > 0.1:
            dealer_alignment = "supportive" if seven.bias == "bullish" else "opposed"
        elif inp.dealer_gamma_exposure < -0.1:
            dealer_alignment = "opposed" if seven.bias == "bullish" else "supportive"
        else:
            dealer_alignment = "neutral"

        # Liquidity comment
        if inp.bid_ask_spread_pct > cfg.spread_veto_threshold:
            liquidity_comment = "wide spread — liquidity suspect"
            veto = True
        elif inp.volume_ratio < 0.7:
            liquidity_comment = "low volume — conviction weak"
        elif inp.volume_ratio > 1.5:
            liquidity_comment = "elevated volume — move appears genuine"
        else:
            liquidity_comment = "liquidity normal"

        return GarakStageOutput(
            trap_risk=round(trap_risk, 4),
            dealer_alignment=dealer_alignment,
            liquidity_comment=liquidity_comment,
            veto=veto,
        )


class _SiskoStage:
    """Issue the final trade directive based on Seven + Garak outputs."""

    def __init__(self, cfg: PersonaConfig) -> None:
        self._cfg = cfg

    def run(self, seven: SevenStageOutput, garak: GarakStageOutput) -> SiskoStageOutput:
        cfg = self._cfg

        # Veto gate
        if garak.veto:
            return SiskoStageOutput(
                directive="no_trade",
                entry_policy="stand aside — Garak veto active",
                invalidation_policy="n/a",
                target_policy="n/a",
                reason=f"veto: trap_risk={garak.trap_risk:.2f} liquidity={garak.liquidity_comment}",
            )

        # Confidence gate
        if seven.confidence < cfg.min_confidence_to_trade:
            return SiskoStageOutput(
                directive="no_trade",
                entry_policy="confidence too low to act",
                invalidation_policy="n/a",
                target_policy="n/a",
                reason=f"confidence={seven.confidence:.2f} below floor={cfg.min_confidence_to_trade}",
            )

        # Neutral bias → no trade
        if seven.bias == "neutral":
            return SiskoStageOutput(
                directive="no_trade",
                entry_policy="no clear directional edge",
                invalidation_policy="n/a",
                target_policy="n/a",
                reason="Seven: neutral bias",
            )

        directive = "long" if seven.bias == "bullish" else "short"

        if directive == "long":
            entry_policy = "enter on continuation above trigger; prefer breakout or pullback to support"
            invalidation_policy = "fail back below trigger level or prior session low"
            target_policy = "scale 50% at first measured move; trail remainder via structure"
        else:
            entry_policy = "enter short on breakdown below trigger; prefer failed rally entry"
            invalidation_policy = "reclaim above breakdown level invalidates thesis"
            target_policy = "scale 50% at first measured target; trail remainder via VWAP"

        return SiskoStageOutput(
            directive=directive,
            entry_policy=entry_policy,
            invalidation_policy=invalidation_policy,
            target_policy=target_policy,
            reason=f"Seven bias={seven.bias} conf={seven.confidence:.2f} garak_trap={garak.trap_risk:.2f}",
        )


class _DataStage:
    """Post-trade / empirical audit over available historical context."""

    def __init__(self, cfg: PersonaConfig) -> None:
        self._cfg = cfg

    def run(self, inp: PersonaPipelineInput, sisko: SiskoStageOutput) -> DataStageOutput:
        cfg = self._cfg
        edge = inp.historical_edge

        if edge >= cfg.high_edge_threshold:
            regime_match = "high"
        elif edge >= cfg.low_edge_threshold:
            regime_match = "moderate"
        else:
            regime_match = "low"

        # Contextual adaptation note
        if inp.realized_vol > 0.30:
            adaptation_note = "elevated vol regime — size down; options extrinsic may be expensive"
        elif inp.realized_vol < 0.12:
            adaptation_note = "compressed vol regime — wider targets; watch for vol expansion catalyst"
        else:
            adaptation_note = f"normal vol environment for {inp.regime_label}"

        # Flag if edge is weak but sisko issued a trade
        review_flag = (edge < cfg.low_edge_threshold) and (sisko.directive != "no_trade")

        return DataStageOutput(
            regime_match=regime_match,
            historical_edge=round(edge, 4),
            adaptation_note=adaptation_note,
            review_flag=review_flag,
        )


class PersonaDecisionPipeline:
    """Orchestrate all four persona stages.

    Usage::

        pipeline = PersonaDecisionPipeline()
        result = pipeline.run(PersonaPipelineInput(symbol="SPY", ...))
        print(result.sisko.directive)
    """

    def __init__(self, config: PersonaConfig | None = None) -> None:
        self._cfg = config or PersonaConfig()
        self._seven = _SevenStage(self._cfg)
        self._garak = _GarakStage(self._cfg)
        self._sisko = _SiskoStage(self._cfg)
        self._data = _DataStage(self._cfg)

    def run(self, inp: PersonaPipelineInput) -> PersonaPipelineResult:
        """Execute the full pipeline and return a structured result."""
        seven = self._seven.run(inp)
        garak = self._garak.run(inp, seven)
        sisko = self._sisko.run(seven, garak)
        data = self._data.run(inp, sisko)
        return PersonaPipelineResult(
            symbol=inp.symbol,
            seven=seven,
            garak=garak,
            sisko=sisko,
            data=data,
        )
