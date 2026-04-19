"""FullRLMPipeline — end-to-end orchestrator for the Regime Locus Matrix engine.

Wires the complete flow in a single composable object:

    bars → FactorPipeline (Kronos + liquidity + GEX/IV)
         → ForecastPipeline (HMM | Markov | plain | probabilistic)
         → KronosRegimeConfidence overlay (default-on, --no-kronos to skip)
         → classify_state_matrix
         → apply_roee_policy
         → (optional) BacktestEngine

Quick start::

    import pandas as pd
    from rlm.core.pipeline import FullRLMPipeline, FullRLMConfig

    bars = pd.read_csv("data/raw/bars_SPY.csv", parse_dates=["timestamp"])
    result = FullRLMPipeline().run(bars)
    print(result.policy_df[["roee_action", "roee_strategy", "roee_size_fraction"]].tail())

Custom config::

    from rlm.core.pipeline import FullRLMPipeline, FullRLMConfig
    from rlm.roee.engine import ROEEConfig

    cfg = FullRLMConfig(
        regime_model="hmm",
        hmm_states=6,
        use_kronos=True,
        roee_config=ROEEConfig(use_dynamic_sizing=True, max_kelly_fraction=0.2),
        run_backtest=True,
    )
    result = FullRLMPipeline(cfg).run(bars, option_chain_df=chain)
    print(result.backtest_metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import pandas as pd

from rlm.data.bars_enrichment import prepare_bars_for_factors
from rlm.factors.cumulative_wyckoff_factors import CumulativeWyckoffFactors
from rlm.factors.hybrid_confluence_factors import HybridConfluenceFactors
from rlm.factors.microstructure_vp_factors import MicrostructureVPFactors
from rlm.features.factors.pipeline import FactorPipeline
from rlm.features.scoring.state_matrix import classify_state_matrix
from rlm.forecasting.engines import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
    HybridProbabilisticForecastPipeline,
)
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.roee.engine import ROEEConfig, apply_roee_policy
from rlm.types.forecast import ForecastConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FullRLMConfig:
    """Unified configuration for the end-to-end RLM pipeline.

    All sub-configs have sensible defaults so callers only need to override
    the knobs they care about.
    """

    # ---- Regime model -------------------------------------------------------
    regime_model: Literal["hmm", "markov", "none"] = "hmm"
    """Which latent-regime model to layer on the forecast.  ``"none"`` runs the
    plain deterministic ``ForecastPipeline``."""
    hmm_states: int = 6
    markov_states: int = 3

    # ---- Forecast -----------------------------------------------------------
    probabilistic: bool = False
    """Add quantile bands (lower/median/upper) via ProbabilisticForecastPipeline."""
    probabilistic_model_path: str | None = None
    drift_gamma_alpha: float = 0.65
    sigma_floor: float = 1e-4
    direction_neutral_threshold: float = 0.3
    move_window: int = 100
    vol_window: int = 100

    # ---- Multi-timeframe ----------------------------------------------------
    mtf: bool = False
    """Augment factors with higher-timeframe overlays via MultiTimeframeEngine."""
    higher_tfs: list[str] = field(default_factory=lambda: ["1W", "1M"])
    mtf_htf_prob_paths: dict[str, str] = field(default_factory=dict)
    """Parquet paths for pre-computed HTF regime probabilities, keyed by TF label."""
    mtf_htf_weights: dict[str, float] = field(default_factory=dict)
    mtf_ltf_weight: float = 0.7
    mtf_regimes: bool = False
    """Blend LTF HMM probs with HTF parquet probs (requires regime_model="hmm")."""

    # ---- Kronos -------------------------------------------------------------
    use_kronos: bool = True
    """Run KronosRegimeConfidence overlay after the forecast step (default-on).
    Set to False for the same effect as passing --no-kronos on the CLI."""

    # ---- ROEE ---------------------------------------------------------------
    roee_config: ROEEConfig = field(default_factory=ROEEConfig)
    strike_increment: float = 1.0

    # ---- Data ---------------------------------------------------------------
    symbol: str = "SPY"
    attach_vix: bool = True
    """Attach ^VIX / ^VVIX via yfinance (requires internet access)."""
    use_intraday_vp: bool = False
    use_cumulative_wyckoff: bool = False
    use_hybrid_confluence: bool = False
    session_type: str = "equity"

    # ---- Nightly overlay ----------------------------------------------------
    nightly_hyperparams_path: str | None = None
    nightly_hyperparams: dict[str, float | int | bool] = field(default_factory=dict)

    # ---- Backtest (optional) ------------------------------------------------
    run_backtest: bool = False
    """Pass True + supply ``option_chain_df`` to run BacktestEngine."""
    initial_capital: float = 100_000.0
    contract_multiplier: int = 100
    quantity_per_trade: int = 1


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Output bundle returned by :meth:`FullRLMPipeline.run`.

    Attributes
    ----------
    factors_df:
        Raw factors, standardized factors, and composite scores (S_D, S_V,
        S_L, S_G) — output of ``FactorPipeline.run()``.
    forecast_df:
        Forecast features with regime annotations and the optional Kronos
        confidence overlay (``kronos_confidence``, ``kronos_regime_agreement``,
        etc.).
    policy_df:
        Full output after ``apply_roee_policy``: includes ``roee_action``,
        ``roee_strategy``, ``roee_size_fraction``, ``vault_triggered``, etc.
    backtest_trades:
        Per-trade log from ``BacktestEngine``; ``None`` when backtest is off.
    backtest_equity:
        Bar-level equity curve from ``BacktestEngine``; ``None`` when off.
    backtest_metrics:
        Summary statistics dict (Sharpe, Sortino, max drawdown, etc.);
        ``None`` when backtest is off.
    """

    factors_df: pd.DataFrame
    forecast_df: pd.DataFrame
    policy_df: pd.DataFrame
    backtest_trades: pd.DataFrame | None = None
    backtest_equity: pd.DataFrame | None = None
    backtest_metrics: dict[str, float] | None = None
    vp_metrics: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FullRLMPipeline:
    """End-to-end RLM orchestrator.

    Parameters
    ----------
    config:
        Pipeline configuration.  Defaults to ``FullRLMConfig()`` which runs
        HMM(6) → Kronos overlay → ROEE with all defaults.

    Example
    -------
    >>> import pandas as pd
    >>> from rlm.core.pipeline import FullRLMPipeline
    >>> bars = pd.read_csv("data/raw/bars_SPY.csv", parse_dates=["timestamp"])
    >>> result = FullRLMPipeline().run(bars)
    >>> result.policy_df.tail(3)
    """

    def __init__(self, config: FullRLMConfig | None = None) -> None:
        self.config: FullRLMConfig = config or FullRLMConfig()

        if self.config.nightly_hyperparams_path:
            nightly_path = Path(self.config.nightly_hyperparams_path)
            if nightly_path.exists():
                nightly = json.loads(nightly_path.read_text(encoding="utf-8"))
                if isinstance(nightly, dict):
                    self.config.nightly_hyperparams.update(nightly)

        nightly = self.config.nightly_hyperparams
        roee_overrides: dict[str, float | int | bool] = {}
        for key, value in nightly.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.roee_config, key):
                roee_overrides[key] = value

        if roee_overrides:
            self.config.roee_config = replace(self.config.roee_config, **roee_overrides)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        bars_df: pd.DataFrame,
        option_chain_df: pd.DataFrame | None = None,
    ) -> PipelineResult:
        """Run the full pipeline end-to-end.

        Parameters
        ----------
        bars_df:
            OHLCV bars with a ``timestamp`` column (or timestamp index).
            Typically from ``data/raw/bars_{SYMBOL}.csv``.
        option_chain_df:
            Normalized option-chain snapshot.  Used for dealer-GEX / skew
            enrichment and, when ``run_backtest=True``, for leg pricing.
            Optional — pipeline degrades gracefully without it.

        Returns
        -------
        PipelineResult
        """
        cfg = self.config

        # 1. Enrich bars with option-chain aggregates and macro series
        df = prepare_bars_for_factors(
            bars_df,
            option_chain_df,
            underlying=cfg.symbol,
            attach_vix=cfg.attach_vix,
        )

        # 2. Factor pipeline (Kronos + liquidity + orderflow + dealer-flow)
        factors_df = FactorPipeline().run(df)

        vp_metrics: pd.DataFrame | None = None
        if cfg.use_intraday_vp and cfg.session_type == "equity":
            vp_metrics = MicrostructureVPFactors(symbol=cfg.symbol).compute(factors_df)
            factors_df = pd.concat([factors_df, vp_metrics], axis=1)
        if cfg.use_cumulative_wyckoff:
            factors_df = pd.concat(
                [factors_df, CumulativeWyckoffFactors().compute(factors_df)], axis=1
            )
        if cfg.use_hybrid_confluence:
            factors_df = pd.concat(
                [factors_df, HybridConfluenceFactors(symbol=cfg.symbol).compute(factors_df)], axis=1
            )

        # 3. Optional multi-timeframe factor augmentation
        if cfg.mtf:
            from rlm.features.factors.multi_timeframe import (
                MultiTimeframeEngine,
                parse_higher_tfs,
            )

            higher_tfs = parse_higher_tfs(",".join(cfg.higher_tfs))
            factors_df = MultiTimeframeEngine(higher_tfs=higher_tfs).augment_factors(df, factors_df)

        # 4. Forecast pipeline (HMM / Markov / plain / probabilistic)
        forecast_df = self._run_forecast(factors_df)

        # 5. Kronos regime-confidence overlay (default-on; set use_kronos=False
        #    to replicate --no-kronos CLI behaviour)
        if cfg.use_kronos:
            from rlm.forecasting.models.kronos import KronosRegimeConfidence

            forecast_df = KronosRegimeConfidence().annotate(forecast_df)

        # 6. State matrix classification → ROEE policy
        state_df = classify_state_matrix(forecast_df)
        policy_df = apply_roee_policy(
            state_df,
            strike_increment=cfg.strike_increment,
            config=cfg.roee_config,
        )

        result = PipelineResult(
            factors_df=factors_df,
            forecast_df=forecast_df,
            policy_df=policy_df,
            vp_metrics=vp_metrics,
        )

        # 7. Optional BacktestEngine (requires option_chain_df)
        if cfg.run_backtest:
            if option_chain_df is None:
                raise ValueError("run_backtest=True requires option_chain_df to be supplied.")
            result = self._run_backtest(result, option_chain_df)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forecast_config(self) -> ForecastConfig:
        cfg = self.config
        return ForecastConfig(
            drift_gamma_alpha=cfg.drift_gamma_alpha,
            sigma_floor=cfg.sigma_floor,
            direction_neutral_threshold=cfg.direction_neutral_threshold,
        )

    def _run_forecast(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        fc = self._forecast_config()
        kw: dict = dict(
            config=fc,
            move_window=cfg.move_window,
            vol_window=cfg.vol_window,
        )

        if cfg.regime_model == "hmm" and cfg.probabilistic:
            return HybridProbabilisticForecastPipeline(
                **kw,
                hmm_config=HMMConfig(n_states=cfg.hmm_states),
                model_path=cfg.probabilistic_model_path,
            ).run(factors_df)

        if cfg.regime_model == "hmm":
            return HybridForecastPipeline(
                **kw,
                hmm_config=HMMConfig(n_states=cfg.hmm_states),
                mtf_regimes=cfg.mtf_regimes,
                mtf_htf_prob_paths=cfg.mtf_htf_prob_paths,
                mtf_htf_weights=cfg.mtf_htf_weights,
                mtf_ltf_weight=cfg.mtf_ltf_weight,
            ).run(factors_df)

        if cfg.regime_model == "markov" and cfg.probabilistic:
            return HybridMarkovForecastPipeline(
                **kw,
                markov_config=MarkovSwitchingConfig(n_states=cfg.markov_states),
                model_path=cfg.probabilistic_model_path,
            ).run(factors_df)

        if cfg.regime_model == "markov":
            return HybridMarkovForecastPipeline(
                **kw,
                markov_config=MarkovSwitchingConfig(n_states=cfg.markov_states),
            ).run(factors_df)

        if cfg.probabilistic:
            return ProbabilisticForecastPipeline(
                **kw,
                model_path=cfg.probabilistic_model_path,
            ).run(factors_df)

        # Plain deterministic
        return ForecastPipeline(**kw).run(factors_df)

    def _run_backtest(
        self,
        result: PipelineResult,
        option_chain_df: pd.DataFrame,
    ) -> PipelineResult:
        from rlm.backtest.engine import BacktestEngine

        cfg = self.config
        engine = BacktestEngine(
            initial_capital=cfg.initial_capital,
            contract_multiplier=cfg.contract_multiplier,
            strike_increment=cfg.strike_increment,
            underlying_symbol=cfg.symbol,
            quantity_per_trade=cfg.quantity_per_trade,
            roee_config=cfg.roee_config,
        )
        trades_df, equity_df, metrics = engine.run(result.policy_df, option_chain_df)
        return PipelineResult(
            factors_df=result.factors_df,
            forecast_df=result.forecast_df,
            policy_df=result.policy_df,
            backtest_trades=trades_df,
            backtest_equity=equity_df,
            backtest_metrics=metrics,
            vp_metrics=result.vp_metrics,
        )
