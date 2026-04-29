from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.core.pipeline import FullRLMConfig
from rlm.features.factors.multi_timeframe import MultiTimeframeEngine
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
from rlm.roee.engine import ROEEConfig
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.types.forecast import ForecastConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    is_window: int = 100
    oos_window: int = 50
    step_size: int = 50
    initial_capital: float = 100_000.0
    strike_increment: float = 1.0
    underlying_symbol: str = "SPY"
    quantity_per_trade: int = 1
    use_dynamic_sizing: bool = False
    max_kelly_fraction: float = 0.25
    regime_adjusted_kelly: bool = True
    high_vol_kelly_multiplier: float = 0.5
    transition_kelly_multiplier: float = 0.75
    calm_trend_kelly_multiplier: float = 1.25
    vault_uncertainty_threshold: float | None = 0.03
    vault_size_multiplier: float = 0.5
    purge_bars: int = 0
    regime_boundary_aware_purge: bool = True
    regime_aware: bool = False
    min_regime_train_samples: int = 20
    log_vp_metrics: bool = True


def _build_walkforward_windows(
    *,
    n_bars: int,
    cfg: WalkForwardConfig,
    regime_keys: pd.Series | None = None,
) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    start = 0
    window_id = 0
    while start + cfg.is_window + cfg.oos_window <= n_bars:
        nominal_is_end = start + cfg.is_window
        effective_is_end = max(start, nominal_is_end - max(int(cfg.purge_bars), 0))
        if (
            cfg.regime_boundary_aware_purge
            and regime_keys is not None
            and 0 < nominal_is_end <= len(regime_keys)
            and effective_is_end > start
        ):
            anchor_idx = nominal_is_end - 1
            anchor_regime = str(regime_keys.iloc[anchor_idx])
            while effective_is_end > start and str(regime_keys.iloc[effective_is_end - 1]) == anchor_regime:
                effective_is_end -= 1
        if effective_is_end <= start:
            effective_is_end = nominal_is_end
        oos_end = nominal_is_end + cfg.oos_window
        windows.append(
            {
                "window_id": window_id,
                "is_start": start,
                "nominal_is_end": nominal_is_end,
                "effective_is_end": effective_is_end,
                "oos_start": nominal_is_end,
                "oos_end": oos_end,
            }
        )
        window_id += 1
        start += cfg.step_size
    return windows


def _expand_training_window_for_regimes(
    *,
    feature_df: pd.DataFrame,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    min_samples: int,
) -> tuple[int, dict[str, int]]:
    coverage = {"coverage_adjusted": 0, "covered_regimes": 0, "required_regimes": 0}
    if not (0 <= train_start < train_end <= len(feature_df)):
        return train_start, coverage
    if not (0 <= test_start < test_end <= len(feature_df)):
        return train_start, coverage

    train = classify_state_matrix(feature_df.iloc[train_start:train_end].copy())
    test = classify_state_matrix(feature_df.iloc[test_start:test_end].copy())
    test_regimes = test["regime_key"].dropna().unique().tolist()
    coverage["required_regimes"] = len(test_regimes)
    if not test_regimes:
        return train_start, coverage

    full_regimes = classify_state_matrix(feature_df.copy())["regime_key"]
    adjusted_start = train_start
    for regime in test_regimes:
        current_count = int((train["regime_key"] == regime).sum())
        if current_count >= min_samples:
            coverage["covered_regimes"] += 1
            continue
        needed = min_samples - current_count
        prior_idx = full_regimes.iloc[:train_start]
        matches = prior_idx[prior_idx == regime]
        if matches.empty:
            continue
        match_positions = feature_df.index.get_indexer(matches.index)
        if len(match_positions) >= needed:
            candidate_start = int(match_positions[-needed])
        else:
            candidate_start = int(match_positions[0])
        adjusted_start = min(adjusted_start, candidate_start)
        coverage["coverage_adjusted"] = 1

    if adjusted_start < train_start:
        train = classify_state_matrix(feature_df.iloc[adjusted_start:train_end].copy())
    coverage["covered_regimes"] = int(
        sum((train["regime_key"] == regime).sum() >= min_samples for regime in test_regimes)
    )
    return adjusted_start, coverage


def run_walkforward(
    *,
    bars: pd.DataFrame,
    option_chain: pd.DataFrame,
    forecast_config: ForecastConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
    use_hmm: bool = False,
    hmm_config: HMMConfig | None = None,
    use_markov: bool = False,
    markov_config: MarkovSwitchingConfig | None = None,
    use_probabilistic: bool = False,
    probabilistic_model_path: str | None = None,
    roee_config: ROEEConfig | None = None,
    hmm_model_dir: Path = Path("models"),
    use_mtf: bool = False,
    higher_tfs: tuple[str, ...] = ("1W", "1M"),
    attach_vix: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = wf_config or WalkForwardConfig()
    fc = forecast_config or ForecastConfig()
    if use_hmm:
        hmm_model_dir.mkdir(parents=True, exist_ok=True)

    bars = bars.sort_index().copy()
    if option_chain is None:
        option_chain = pd.DataFrame()
    # Local import avoids circular import: rlm.datasets.backtest_data → walkforward.
    from rlm.data.bars_enrichment import prepare_bars_for_factors

    bars = prepare_bars_for_factors(
        bars,
        option_chain,
        underlying=cfg.underlying_symbol,
        attach_vix=attach_vix,
    )
    all_equity = []
    all_trades = []
    window_summaries = []
    hmm_model_dir.mkdir(parents=True, exist_ok=True)
    mtf_engine = MultiTimeframeEngine(higher_tfs=higher_tfs) if use_mtf else None

    global_regimes = classify_state_matrix(FactorPipeline().run(bars.copy()))["regime_key"]
    windows = _build_walkforward_windows(n_bars=len(bars), cfg=cfg, regime_keys=global_regimes)
    for window in windows:
        window_id = int(window["window_id"])
        nominal_is_start = int(window["is_start"])
        effective_is_end = int(window["effective_is_end"])
        oos_start = int(window["oos_start"])
        oos_end = int(window["oos_end"])

        joined = bars.iloc[nominal_is_start:oos_end].copy()
        feature_df = FactorPipeline().run(joined)
        if mtf_engine is not None:
            feature_df = mtf_engine.augment_factors(joined, feature_df)

        train_start_in_joined = 0
        train_end_in_joined = max(effective_is_end - nominal_is_start, 0)
        test_start_in_joined = oos_start - nominal_is_start
        test_end_in_joined = oos_end - nominal_is_start

        regime_meta = {
            "coverage_adjusted": 0,
            "covered_regimes": 0,
            "required_regimes": 0,
        }
        if cfg.regime_aware:
            train_start_in_joined, regime_meta = _expand_training_window_for_regimes(
                feature_df=feature_df,
                train_start=train_start_in_joined,
                train_end=train_end_in_joined,
                test_start=test_start_in_joined,
                test_end=test_end_in_joined,
                min_samples=max(int(cfg.min_regime_train_samples), 1),
            )

        is_bars = joined.iloc[train_start_in_joined:train_end_in_joined].copy()
        oos_bars = joined.iloc[test_start_in_joined:test_end_in_joined].copy()

        if use_hmm and use_probabilistic:
            forecast_pipeline = HybridProbabilisticForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
                hmm_config=hmm_config or HMMConfig(),
                model_path=probabilistic_model_path,
            )
            train_mask = feature_df.index.isin(is_bars.index)
            feature_df = forecast_pipeline.run(feature_df, train_mask=pd.Series(train_mask, index=feature_df.index))

            hmm_path = hmm_model_dir / f"hmm_fold_{window_id}.pkl"
            forecast_pipeline.hmm.save(hmm_path)
        elif use_markov and use_probabilistic:
            feature_df = HybridMarkovForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
                markov_config=markov_config or MarkovSwitchingConfig(),
                model_path=probabilistic_model_path,
            ).run(
                feature_df,
                train_mask=pd.Series(feature_df.index.isin(is_bars.index), index=feature_df.index),
            )
        elif use_hmm:
            forecast_pipeline = HybridForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
                hmm_config=hmm_config or HMMConfig(),
            )
            train_mask = feature_df.index.isin(is_bars.index)
            feature_df = forecast_pipeline.run(feature_df, train_mask=pd.Series(train_mask, index=feature_df.index))

            hmm_path = hmm_model_dir / f"hmm_fold_{window_id}.pkl"
            forecast_pipeline.hmm.save(hmm_path)
        elif use_markov:
            feature_df = HybridMarkovForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
                markov_config=markov_config or MarkovSwitchingConfig(),
                model_path=None,
            ).run(
                feature_df,
                train_mask=pd.Series(feature_df.index.isin(is_bars.index), index=feature_df.index),
            )
        elif use_probabilistic:
            feature_df = ProbabilisticForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
                model_path=probabilistic_model_path,
            ).run(feature_df)
        else:
            feature_df = ForecastPipeline(
                config=fc,
                move_window=cfg.is_window,
                vol_window=cfg.is_window,
            ).run(feature_df)

        feature_df = classify_state_matrix(feature_df)
        feature_df = attach_regime_safety_columns(
            feature_df,
            min_regime_train_samples=cfg.min_regime_train_samples,
            purge_bars=cfg.purge_bars,
        )
        oos_features = feature_df.loc[oos_bars.index].copy()
        if option_chain is None or option_chain.empty or "timestamp" not in option_chain.columns:
            oos_chain = pd.DataFrame()
        else:
            oos_chain = option_chain[option_chain["timestamp"].isin(oos_features.index)].copy()

        effective_roee_config = roee_config or ROEEConfig(
            use_dynamic_sizing=(cfg.use_dynamic_sizing or use_hmm or use_markov),
            max_kelly_fraction=cfg.max_kelly_fraction,
            regime_adjusted_kelly=cfg.regime_adjusted_kelly,
            high_vol_kelly_multiplier=cfg.high_vol_kelly_multiplier,
            transition_kelly_multiplier=cfg.transition_kelly_multiplier,
            calm_trend_kelly_multiplier=cfg.calm_trend_kelly_multiplier,
            vault_uncertainty_threshold=cfg.vault_uncertainty_threshold,
            vault_size_multiplier=cfg.vault_size_multiplier,
        )
        effective_roee_config = replace(
            effective_roee_config,
            min_regime_train_samples=max(int(cfg.min_regime_train_samples), 0),
            purge_bars=max(int(cfg.purge_bars), 0),
        )

        engine = BacktestEngine(
            initial_capital=cfg.initial_capital,
            strike_increment=cfg.strike_increment,
            underlying_symbol=cfg.underlying_symbol,
            quantity_per_trade=cfg.quantity_per_trade,
            roee_config=effective_roee_config,
        )

        equity_frame, trades_frame, summary = engine.run(oos_features, oos_chain)

        if not equity_frame.empty:
            eq = equity_frame.copy()
            eq["window_id"] = window_id
            all_equity.append(eq)

        if not trades_frame.empty:
            tr = trades_frame.copy()
            tr["window_id"] = window_id
            all_trades.append(tr)

        summary_row = {
            "window_id": window_id,
            "is_start": str(is_bars.index[0]),
            "is_end": str(is_bars.index[-1]),
            "oos_start": str(oos_bars.index[0]),
            "oos_end": str(oos_bars.index[-1]),
            "nominal_is_start": str(joined.index[0]),
            "nominal_is_end": str(joined.index[max(train_end_in_joined - 1, 0)]),
            "purge_bars": int(cfg.purge_bars),
            "regime_aware": bool(cfg.regime_aware),
            "regime_boundary_aware_purge": bool(cfg.regime_boundary_aware_purge),
            "unsafe_oos_bars": int((~oos_features["regime_safety_ok"]).sum()),
            "last_oos_regime_train_samples": int(oos_features["regime_train_sample_count"].iloc[-1]),
            "min_oos_regime_train_samples": int(oos_features["regime_train_sample_count"].min()),
            "regime_safety_fraction": round(float(oos_features["regime_safety_ok"].mean()), 3),
            "regime_safety_passed": float(oos_features["regime_safety_ok"].mean()) >= 0.70,
            **regime_meta,
        }
        if cfg.log_vp_metrics:
            balance_ratio = (
                float((oos_features["vp_auction_state"].astype(str).str.lower() == "balance").mean())
                if "vp_auction_state" in oos_features.columns and len(oos_features) > 0
                else float("nan")
            )
            summary_row.update(
                {
                    "avg_auction_state_balance_ratio": balance_ratio,
                    "avg_wyckoff_divergence": (
                        float(pd.to_numeric(oos_features["cumulative_wyckoff_score"], errors="coerce").mean())
                        if "cumulative_wyckoff_score" in oos_features.columns
                        else float("nan")
                    ),
                    "avg_hybrid_strength": (
                        float(pd.to_numeric(oos_features["vp_hybrid_strength_max"], errors="coerce").mean())
                        if "vp_hybrid_strength_max" in oos_features.columns
                        else float("nan")
                    ),
                    "eighty_percent_rule_hit_rate": (
                        float(
                            pd.to_numeric(oos_features["vp_eighty_percent_signal"], errors="coerce").fillna(0.0).mean()
                        )
                        if "vp_eighty_percent_signal" in oos_features.columns
                        else float("nan")
                    ),
                }
            )
        summary_row.update(summary)
        window_summaries.append(summary_row)

    equity_df = pd.concat(all_equity).sort_index() if all_equity else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(window_summaries)

    return equity_df, trades_df, summary_df


@dataclass
class WalkForwardResult:
    """Structured output from :class:`WalkForwardEngine`."""

    equity_df: pd.DataFrame
    trades_df: pd.DataFrame
    summary_df: pd.DataFrame

    @property
    def window_results(self) -> list[dict[str, Any]]:
        if self.summary_df.empty:
            return []
        return self.summary_df.to_dict(orient="records")


class WalkForwardEngine:
    """Run walk-forward OOS backtests with :func:`run_walkforward` and a :class:`FullRLMConfig`."""

    def __init__(
        self,
        pipeline_config: FullRLMConfig,
        wf_config: WalkForwardConfig | None = None,
        *,
        hmm_model_dir: Path | None = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.wf_config = wf_config or WalkForwardConfig()
        self.hmm_model_dir = hmm_model_dir

    def run(self, bars: pd.DataFrame, option_chain: pd.DataFrame | None) -> WalkForwardResult:
        cfg = self.pipeline_config
        wf = replace(
            self.wf_config,
            initial_capital=cfg.initial_capital,
            underlying_symbol=cfg.symbol,
            strike_increment=cfg.strike_increment,
        )
        fc = ForecastConfig(
            drift_gamma_alpha=cfg.drift_gamma_alpha,
            sigma_floor=cfg.sigma_floor,
            direction_neutral_threshold=cfg.direction_neutral_threshold,
        )
        use_hmm = cfg.regime_model == "hmm"
        use_markov = cfg.regime_model == "markov"
        hmm_c = (
            HMMConfig(
                n_states=cfg.hmm_states,
                transition_pseudocount=cfg.hmm_transition_pseudocount,
            )
            if use_hmm
            else None
        )
        vp_cfg = cfg.volume_profile
        markov_c: MarkovSwitchingConfig | None
        if use_markov:
            markov_c = MarkovSwitchingConfig(
                n_states=cfg.markov_states,
                transition_pseudocount=cfg.markov_transition_pseudocount,
                use_intraday_vp_features=vp_cfg.enabled and vp_cfg.intraday_enabled,
                use_wyckoff_features=vp_cfg.enabled and vp_cfg.wyckoff_enabled,
                use_confluence_features=vp_cfg.enabled and vp_cfg.confluence_enabled,
            )
        else:
            markov_c = None
        hmm_dir = self.hmm_model_dir if self.hmm_model_dir is not None else Path("models")
        higher = tuple(cfg.higher_tfs) if cfg.higher_tfs else ("1W", "1M")
        oc = option_chain if option_chain is not None else pd.DataFrame()
        eq, tr, sm = run_walkforward(
            bars=bars,
            option_chain=oc,
            forecast_config=fc,
            wf_config=wf,
            use_hmm=use_hmm,
            hmm_config=hmm_c,
            use_markov=use_markov,
            markov_config=markov_c,
            use_probabilistic=cfg.probabilistic,
            probabilistic_model_path=cfg.probabilistic_model_path,
            roee_config=cfg.roee_config,
            hmm_model_dir=hmm_dir,
            use_mtf=cfg.mtf,
            higher_tfs=higher,
            attach_vix=cfg.attach_vix,
        )
        return WalkForwardResult(equity_df=eq, trades_df=tr, summary_df=sm)
