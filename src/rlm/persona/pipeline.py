"""PersonaDecisionPipeline — lightweight four-stage interpretation layer."""

from __future__ import annotations

import pandas as pd

from rlm.persona.config import PersonaConfig
from rlm.persona.models import PersonaInputs, PersonaPipelineResult
from rlm.persona.stages import run_data, run_garak, run_seven, run_sisko


class PersonaDecisionPipeline:
    """Orchestrate the Seven → Garak → Sisko → Data interpretation chain.

    This pipeline is an *interpreter* over existing RLM outputs.  It does not
    re-compute factors or regimes; it reads the last bar of a ``PipelineResult``
    and produces a structured trade-interpretation artefact.

    Parameters
    ----------
    config:
        Threshold overrides.  Defaults are conservative and suitable for most
        use-cases; only override what you need.

    Example
    -------
    ::

        from rlm.core.pipeline import FullRLMPipeline
        from rlm.persona.pipeline import PersonaDecisionPipeline

        rlm_result = FullRLMPipeline().run(bars_df)
        persona = PersonaDecisionPipeline().run(rlm_result)
        print(persona.sisko.directive)          # "long" | "short" | "no_trade"
        print(persona.to_dict())                # full JSON-serialisable dict
    """

    def __init__(self, config: PersonaConfig | None = None) -> None:
        self.config: PersonaConfig = config or PersonaConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, result: object) -> PersonaPipelineResult:
        """Run all four stages over a ``PipelineResult``.

        Parameters
        ----------
        result:
            Output from ``FullRLMPipeline.run()``.  Typed as ``object`` to
            avoid a circular import; at runtime must expose ``factors_df``,
            ``forecast_df``, ``policy_df``, and optionally ``backtest_metrics``.

        Returns
        -------
        PersonaPipelineResult
        """
        inputs = self._extract_inputs(result)
        seven = run_seven(inputs, self.config)
        garak = run_garak(inputs, seven, self.config)
        sisko = run_sisko(inputs, seven, garak, self.config)
        data = run_data(inputs, sisko, self.config)
        return PersonaPipelineResult(seven=seven, garak=garak, sisko=sisko, data=data)

    def run_from_inputs(self, inputs: PersonaInputs) -> PersonaPipelineResult:
        """Run the pipeline from a pre-built :class:`PersonaInputs`.

        Useful for testing or when you already have scalar inputs available.
        """
        seven = run_seven(inputs, self.config)
        garak = run_garak(inputs, seven, self.config)
        sisko = run_sisko(inputs, seven, garak, self.config)
        data = run_data(inputs, sisko, self.config)
        return PersonaPipelineResult(seven=seven, garak=garak, sisko=sisko, data=data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_inputs(result: object) -> PersonaInputs:
        """Pull scalar values from the last bar of a PipelineResult."""

        factors_df: pd.DataFrame = getattr(result, "factors_df", pd.DataFrame())
        forecast_df: pd.DataFrame = getattr(result, "forecast_df", pd.DataFrame())
        policy_df: pd.DataFrame = getattr(result, "policy_df", pd.DataFrame())
        backtest_metrics: dict | None = getattr(result, "backtest_metrics", None)

        def _last(df: pd.DataFrame, col: str, default: object = None) -> object:
            if not df.empty and col in df.columns:
                val = df.iloc[-1][col]
                return None if pd.isna(val) else val
            return default

        # Factor scores from factors_df
        s_d = float(_last(factors_df, "S_D") or 0.0)
        s_v = float(_last(factors_df, "S_V") or 0.0)
        s_l = float(_last(factors_df, "S_L") or 0.0)
        s_g = float(_last(factors_df, "S_G") or 0.0)

        # Regime labels from policy_df (populated by classify_state_matrix + apply_roee_policy)
        direction_regime = str(_last(policy_df, "direction_regime") or "neutral")
        volatility_regime = str(_last(policy_df, "volatility_regime") or "neutral")
        liquidity_regime = str(_last(policy_df, "liquidity_regime") or "neutral")
        dealer_flow_regime = str(_last(policy_df, "dealer_flow_regime") or "neutral")

        # HMM confidence from forecast_df; fall back to kronos_confidence then 0.5
        hmm_conf = _last(forecast_df, "hmm_confidence")
        if hmm_conf is None:
            hmm_conf = _last(forecast_df, "kronos_confidence")
        hmm_confidence = float(hmm_conf if hmm_conf is not None else 0.5)

        roee_action = _last(policy_df, "roee_action")

        return PersonaInputs(
            s_d=s_d,
            s_v=s_v,
            s_l=s_l,
            s_g=s_g,
            direction_regime=direction_regime,
            volatility_regime=volatility_regime,
            liquidity_regime=liquidity_regime,
            dealer_flow_regime=dealer_flow_regime,
            hmm_confidence=hmm_confidence,
            roee_action=None if roee_action is None else str(roee_action),
            backtest_metrics=backtest_metrics,
        )
