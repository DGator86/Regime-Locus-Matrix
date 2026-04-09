from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from rlm.backtest.engine import BacktestEngine
from rlm.factors.pipeline import FactorPipeline
from rlm.factors.multi_timeframe import MultiTimeframeEngine
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
    HybridProbabilisticForecastPipeline,
)
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.roee.pipeline import ROEEConfig
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.scoring.state_matrix import classify_state_matrix
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
    regime_aware: bool = False
    min_regime_train_samples: int = 20


def _build_walkforward_windows(
    *,
    n_bars: int,
    cfg: WalkForwardConfig,
) -> list[dict[str, int]]:
    windows: list[dict[str, int]] = []
    start = 0
    window_id = 0
    while start + cfg.is_window + cfg.oos_window <= n_bars:
        nominal_is_end = start + cfg.is_window
        effective_is_end = max(start, nominal_is_end - max(int(cfg.purge_bars), 0))
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = wf_config or WalkForwardConfig()
    fc = forecast_config or ForecastConfig()
    if use_hmm:
        hmm_model_dir.mkdir(parents=True, exist_ok=True)

    bars = bars.sort_index().copy()
    # Local import avoids circular import: rlm.datasets.backtest_data → walkforward.
    from rlm.datasets.bars_enrichment import prepare_bars_for_factors

    bars = prepare_bars_for_factors(
        bars,
        option_chain,
        underlying=cfg.underlying_symbol,
        attach_vix=True,
    )
    all_equity = []
    all_trades = []
    window_summaries = []
    hmm_model_dir.mkdir(parents=True, exist_ok=True)
    mtf_engine = MultiTimeframeEngine(higher_tfs=higher_tfs) if use_mtf else None

    windows = _build_walkforward_windows(n_bars=len(bars), cfg=cfg)
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
            feature_df = forecast_pipeline.run(
                feature_df, train_mask=pd.Series(train_mask, index=feature_df.index)
            )

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
            feature_df = forecast_pipeline.run(
                feature_df, train_mask=pd.Series(train_mask, index=feature_df.index)
            )

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
            "unsafe_oos_bars": int((~oos_features["regime_safety_ok"]).sum()),
            "last_oos_regime_train_samples": int(
                oos_features["regime_train_sample_count"].iloc[-1]
            ),
            "min_oos_regime_train_samples": int(oos_features["regime_train_sample_count"].min()),
            "regime_safety_passed": bool(oos_features["regime_safety_ok"].all()),
            **regime_meta,
        }
        summary_row.update(summary)
        window_summaries.append(summary_row)

    equity_df = pd.concat(all_equity).sort_index() if all_equity else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(window_summaries)

    return equity_df, trades_df, summary_df
