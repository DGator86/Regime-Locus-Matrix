from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from rlm.backtest.fills import FillConfig
from rlm.backtest.lifecycle import (
    ExpiryLiquidationPolicy,
    LifecycleConfig,
    is_at_or_past_expiry,
    should_close_for_max_holding,
    should_force_close_before_expiry,
)
from rlm.backtest.portfolio import Portfolio
from rlm.backtest.slippage import SlippageConfig
from rlm.data.option_chain import normalize_option_chain, select_nearest_expiry_slice
from rlm.roee.chain_match import match_legs_to_chain
from rlm.roee.decision import select_trade_for_row
from rlm.roee.exits import (
    should_exit_for_profit,
    should_exit_for_regime_flip,
    should_exit_for_zone_breach,
    should_exit_for_stop_loss,
    should_exit_for_time_stop,
)
from rlm.roee.engine import ROEEConfig
from rlm.roee.regime_safety import attach_regime_safety_columns
from rlm.features.scoring.state_matrix import classify_state_matrix

# Alias for tests and external monkeypatching (backtests call this once per bar).
decide_trade_for_bar = select_trade_for_row


@dataclass(frozen=True)
class MTFWeightConfig:
    fast_weight: float = 0.6
    medium_weight: float = 0.3
    slow_weight: float = 0.1
    medium_window: int = 3
    slow_window: int = 10


@dataclass(frozen=True)
class MonteCarloBootstrapConfig:
    n_paths: int = 500
    sample_frac: float = 1.0
    random_seed: int = 42


@dataclass(frozen=True)
class GapRiskStressConfig:
    downside_gap_bps: tuple[float, ...] = (0.0, 75.0, 150.0, 300.0)
    regime_multipliers: dict[str, float] | None = None


@dataclass(frozen=True)
class HyperOptConfig:
    n_trials: int = 50
    timeout_seconds: int | None = None
    metric: str = "sharpe"


@dataclass(frozen=True)
class BacktestConfig:
    use_vp_gating: bool = False
    wyckoff_threshold: float = 0.7
    balance_haircut: float = 0.5
    eighty_percent_boost: float = 0.2
    hybrid_strength_scaling: bool = True
    gex_confluence_enabled: bool = True


class BacktestEngine:
    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        contract_multiplier: int = 100,
        strike_increment: float = 1.0,
        underlying_symbol: str = "SPY",
        quantity_per_trade: int = 1,
        fill_config: FillConfig | None = None,
        lifecycle_config: LifecycleConfig | None = None,
        roee_config: ROEEConfig | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.lifecycle_config = lifecycle_config or LifecycleConfig()
        self.roee_config = roee_config
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            contract_multiplier=contract_multiplier,
            lifecycle_config=self.lifecycle_config,
        )
        self.strike_increment = strike_increment
        self.underlying_symbol = underlying_symbol
        self.quantity_per_trade = quantity_per_trade
        self.fill_config = fill_config or FillConfig(contract_multiplier=contract_multiplier)
        self.config = config or BacktestConfig()

    def run(
        self,
        feature_df: pd.DataFrame,
        option_chain_df: pd.DataFrame,
        *,
        mtf_config: MTFWeightConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        equity_frame, trades_frame, summary, _ = self.run_with_robustness(
            feature_df,
            option_chain_df,
            mtf_config=mtf_config,
            monte_carlo=None,
            gap_risk=None,
        )
        return equity_frame, trades_frame, summary

    def run_with_robustness(
        self,
        feature_df: pd.DataFrame,
        option_chain_df: pd.DataFrame,
        *,
        mtf_config: MTFWeightConfig | None = None,
        monte_carlo: MonteCarloBootstrapConfig | None = None,
        gap_risk: GapRiskStressConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, Any]]:
        chain = normalize_option_chain(option_chain_df)
        features = self._apply_mtf_weights(feature_df.copy(), mtf_config=mtf_config)
        features = classify_state_matrix(features)
        rc = self.roee_config or ROEEConfig()
        features = attach_regime_safety_columns(
            features,
            min_regime_train_samples=rc.min_regime_train_samples,
            purge_bars=rc.purge_bars,
        )

        ch = chain.loc[chain["underlying"] == self.underlying_symbol].copy()
        chain_by_ts: dict[pd.Timestamp, pd.DataFrame] = {}
        for tkey, grp in ch.groupby("timestamp", sort=False):
            chain_by_ts[pd.Timestamp(tkey)] = grp.copy()

        for bar_index, (ts, row) in enumerate(features.iterrows()):
            traded_this_bar = False
            row_chain = chain_by_ts.get(pd.Timestamp(ts), pd.DataFrame()).copy()

            if not row_chain.empty:
                self.portfolio.revalue_open_positions(
                    chain_snapshot=row_chain,
                    fill_config=self.fill_config,
                )

            self._process_exits(ts, row, bar_index)

            if row_chain.empty:
                self.portfolio.mark_equity(ts)
                continue

            decision = decide_trade_for_bar(
                row,
                strike_increment=self.strike_increment,
                hmm_confidence_threshold=rc.hmm_confidence_threshold,
                hmm_sizing_multiplier=rc.sizing_multiplier,
                hmm_transition_penalty=rc.transition_penalty,
                use_dynamic_sizing=rc.use_dynamic_sizing,
                vol_target=rc.vol_target,
                max_kelly_fraction=rc.max_kelly_fraction,
                max_capital_fraction=rc.max_capital_fraction,
                regime_adjusted_kelly=rc.regime_adjusted_kelly,
                high_vol_kelly_multiplier=rc.high_vol_kelly_multiplier,
                transition_kelly_multiplier=rc.transition_kelly_multiplier,
                calm_trend_kelly_multiplier=rc.calm_trend_kelly_multiplier,
                vault_uncertainty_threshold=rc.vault_uncertainty_threshold,
                vault_size_multiplier=rc.vault_size_multiplier,
                regime_train_sample_count=int(row.get("regime_train_sample_count", 0) or 0),
                min_regime_train_samples=rc.min_regime_train_samples,
                regime_purge_bars=rc.purge_bars,
                use_volume_profile_gating=self.config.use_vp_gating or rc.vp_gating_enabled,
                wyckoff_threshold=self.config.wyckoff_threshold,
                balance_haircut=self.config.balance_haircut,
                eighty_percent_boost=self.config.eighty_percent_boost,
                hybrid_strength_scaling=self.config.hybrid_strength_scaling,
                gex_confluence_enabled=self.config.gex_confluence_enabled,
            )

            if decision.action == "enter" and not (
                self.lifecycle_config.one_trade_per_bar and traded_this_bar
            ):
                # --- Portfolio Manager Enforcement ---
                symbol_cap = rc.max_positions_per_symbol
                total_cap = rc.max_total_positions
                
                has_symbol_pos = self.portfolio.has_position_for_symbol(self.underlying_symbol)
                total_pos_count = self.portfolio.total_position_count()
                
                if has_symbol_pos and symbol_cap > 0:
                    decision.action = "skip"
                    decision.rationale = f"Symbol cap reached: {self.underlying_symbol}"
                elif total_pos_count >= total_cap and total_cap > 0:
                    decision.action = "skip"
                    decision.rationale = f"Portfolio cap reached: {total_pos_count}/{total_cap}"
                
            if decision.action == "enter" and not (
                self.lifecycle_config.one_trade_per_bar and traded_this_bar
            ):
                dte_min = int(decision.candidate.target_dte_min) if decision.candidate else 20
                dte_max = int(decision.candidate.target_dte_max) if decision.candidate else 45
                chain_slice = select_nearest_expiry_slice(
                    row_chain, dte_min=dte_min, dte_max=dte_max
                )

                if not chain_slice.empty:
                    matched_decision = match_legs_to_chain(
                        decision=decision, chain_slice=chain_slice
                    )
                    if matched_decision.action == "enter":
                        opened_id = self.portfolio.open_from_decision(
                            timestamp=pd.Timestamp(ts),
                            underlying_symbol=self.underlying_symbol,
                            underlying_price=float(row["close"]),
                            decision=matched_decision,
                            quantity=self.quantity_per_trade,
                            fill_config=self.fill_config,
                            bar_index=bar_index,
                        )
                        traded_this_bar = opened_id is not None

            self.portfolio.mark_equity(ts)

        equity_frame = self.portfolio.equity_frame()
        trades_frame = self.portfolio.closed_trades_frame()

        from rlm.backtest.metrics import summarize_backtest

        summary = summarize_backtest(equity_frame, trades_frame)
        diagnostics = self._build_robustness_diagnostics(
            equity_frame=equity_frame,
            trades_frame=trades_frame,
            features=features,
            monte_carlo=monte_carlo,
            gap_risk=gap_risk,
        )
        return equity_frame, trades_frame, summary, diagnostics

    def _apply_mtf_weights(
        self, feature_df: pd.DataFrame, *, mtf_config: MTFWeightConfig | None
    ) -> pd.DataFrame:
        if mtf_config is None:
            return feature_df
        out = feature_df.copy()
        cols = [c for c in ["S_D", "S_V", "S_L", "S_G"] if c in out.columns]
        if not cols:
            return out
        w_fast = max(float(mtf_config.fast_weight), 0.0)
        w_med = max(float(mtf_config.medium_weight), 0.0)
        w_slow = max(float(mtf_config.slow_weight), 0.0)
        denom = max(w_fast + w_med + w_slow, 1e-9)
        for col in cols:
            fast = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            med = fast.rolling(max(int(mtf_config.medium_window), 1), min_periods=1).mean()
            slow = fast.rolling(max(int(mtf_config.slow_window), 1), min_periods=1).mean()
            out[col] = (w_fast * fast + w_med * med + w_slow * slow) / denom
        return out

    def _build_robustness_diagnostics(
        self,
        *,
        equity_frame: pd.DataFrame,
        trades_frame: pd.DataFrame,
        features: pd.DataFrame,
        monte_carlo: MonteCarloBootstrapConfig | None,
        gap_risk: GapRiskStressConfig | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if monte_carlo is not None and not trades_frame.empty and "pnl" in trades_frame.columns:
            out["monte_carlo"] = self._monte_carlo_by_regime(
                trades_frame=trades_frame,
                n_paths=max(int(monte_carlo.n_paths), 1),
                sample_frac=float(monte_carlo.sample_frac),
                random_seed=int(monte_carlo.random_seed),
            )
        if gap_risk is not None and not equity_frame.empty:
            out["gap_risk"] = self._gap_risk_stress(
                equity_frame=equity_frame,
                features=features,
                gap_risk=gap_risk,
            )
        return out

    def _monte_carlo_by_regime(
        self,
        *,
        trades_frame: pd.DataFrame,
        n_paths: int,
        sample_frac: float,
        random_seed: int,
    ) -> dict[str, float]:
        rng = np.random.default_rng(random_seed)
        regime_col = "entry_regime" if "entry_regime" in trades_frame.columns else "regime_key"
        if regime_col not in trades_frame.columns:
            regime_col = "_default"
            trades = trades_frame.copy()
            trades[regime_col] = "all"
        else:
            trades = trades_frame

        returns_by_regime = {
            str(regime): grp["pnl"].dropna().astype(float).values
            for regime, grp in trades.groupby(regime_col)
            if not grp.empty
        }
        if not returns_by_regime:
            return {}

        combined_paths: list[float] = []
        for _ in range(n_paths):
            pnl = 0.0
            for values in returns_by_regime.values():
                if len(values) == 0:
                    continue
                n_draw = max(1, int(round(len(values) * max(min(sample_frac, 1.0), 0.05))))
                sampled = rng.choice(values, size=n_draw, replace=True)
                pnl += float(sampled.sum())
            combined_paths.append(pnl)

        arr = np.array(combined_paths, dtype=float)
        return {
            "mc_expected_pnl": float(arr.mean()),
            "mc_p05_pnl": float(np.quantile(arr, 0.05)),
            "mc_p50_pnl": float(np.quantile(arr, 0.50)),
            "mc_p95_pnl": float(np.quantile(arr, 0.95)),
        }

    def _gap_risk_stress(
        self,
        *,
        equity_frame: pd.DataFrame,
        features: pd.DataFrame,
        gap_risk: GapRiskStressConfig,
    ) -> dict[str, float]:
        gap_levels = tuple(float(x) for x in gap_risk.downside_gap_bps)
        multipliers = gap_risk.regime_multipliers or {}
        eq = equity_frame["equity"].astype(float)
        stress: dict[str, float] = {}
        for bps in gap_levels:
            shocked = eq.copy()
            pct = max(bps, 0.0) / 10_000.0
            if pct <= 0:
                stress[f"final_equity_gap_{int(bps)}bps"] = float(shocked.iloc[-1])
                continue
            if "regime_key" in features.columns:
                penalty = features["regime_key"].astype(str).map(multipliers).fillna(1.0)
                penalty = penalty.reindex(shocked.index).fillna(1.0)
            else:
                penalty = pd.Series(1.0, index=shocked.index)
            shocked = shocked * (1.0 - pct * penalty.clip(lower=0.0))
            stress[f"final_equity_gap_{int(bps)}bps"] = float(shocked.iloc[-1])
        return stress

    def _process_exits(self, ts: pd.Timestamp, row: pd.Series, bar_index: int) -> None:
        to_close: list[tuple[str, str]] = []
        to_settle: list[str] = []

        policy = self.lifecycle_config.expiry_liquidation_policy

        for position_id, pos in self.portfolio.open_positions.items():
            if should_close_for_max_holding(
                entry_bar_index=pos.entry_bar_index,
                current_bar_index=bar_index,
                config=self.lifecycle_config,
            ):
                to_close.append((position_id, "max_holding"))
                continue

            at_expiry = is_at_or_past_expiry(
                timestamp=pd.Timestamp(ts),
                expiry=pos.expiry,
            )

            if policy == ExpiryLiquidationPolicy.SETTLE_AT_EXPIRY:
                if at_expiry:
                    to_settle.append(position_id)
                    continue
            else:
                if should_force_close_before_expiry(
                    timestamp=pd.Timestamp(ts),
                    expiry=pos.expiry,
                    config=self.lifecycle_config,
                ):
                    to_close.append((position_id, "forced_pre_expiry"))
                    continue

                if self.lifecycle_config.close_at_expiry_if_open and at_expiry:
                    to_close.append((position_id, "expiry_close"))
                    continue

            pricing_ok = bool(pos.metadata.get("reprice_ok", True))

            pnl_pct = pos.pnl_pct()

            if pricing_ok and should_exit_for_profit(
                pnl_pct=pnl_pct, target_profit_pct=pos.target_profit_pct
            ):
                to_close.append((position_id, "profit_target"))
                continue

            # Hard Stop Loss
            if pricing_ok and should_exit_for_stop_loss(
                pnl_pct=pnl_pct, stop_loss_pct=self.roee_config.hard_stop_loss_pct if self.roee_config else -0.50
            ):
                to_close.append((position_id, "stop_loss"))
                continue

            # Time Stop (DTE)
            from rlm.data.option_chain import calculate_dte_from_expiry
            dte = calculate_dte_from_expiry(pos.expiry, ts) if pos.expiry else 999
            if pricing_ok and should_exit_for_time_stop(
                dte_remaining=dte, min_dte_threshold=self.roee_config.force_exit_dte if self.roee_config else 2
            ):
                to_close.append((position_id, "time_stop"))
                continue

            if pricing_ok and should_exit_for_zone_breach(
                realized_price=float(row["close"]),
                lower_1s=float(row["lower_1s"]),
                upper_1s=float(row["upper_1s"]),
            ):
                to_close.append((position_id, "zone_breach"))
                continue

            bars_held = (
                bar_index - pos.entry_bar_index
                if pos.entry_bar_index is not None
                else self.lifecycle_config.min_hold_bars
            )
            if (
                pricing_ok
                and bars_held >= self.lifecycle_config.min_hold_bars
                and should_exit_for_regime_flip(
                    entry_regime_key=pos.regime_key,
                    current_regime_key=str(row["regime_key"]),
                )
            ):
                to_close.append((position_id, "regime_flip"))
                continue

        for position_id, reason in to_close:
            self.portfolio.close_position(
                position_id=position_id,
                timestamp_exit=pd.Timestamp(ts),
                underlying_price=float(row["close"]),
                exit_reason=reason,
            )

        for position_id in to_settle:
            self.portfolio.expiry_settle_position(
                position_id=position_id,
                timestamp_exit=pd.Timestamp(ts),
                underlying_price=float(row["close"]),
            )


class BacktestHyperparameterOptimizer:
    """Optuna-powered Bayesian search across regime/forecast/cost/MTF parameters."""

    def __init__(self, config: HyperOptConfig | None = None) -> None:
        self.config = config or HyperOptConfig()

    def optimize(
        self,
        *,
        feature_builder: Callable[[dict[str, float]], pd.DataFrame],
        option_chain_df: pd.DataFrame,
        engine_builder: Callable[[dict[str, float]], BacktestEngine],
    ) -> tuple[dict[str, float], dict[str, float], Any]:
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Optuna is required for --optuna-trials. Install optuna.") from exc

        direction = "maximize"
        metric = self.config.metric

        def objective(trial: Any) -> float:
            params: dict[str, float] = {
                "hmm_states": float(trial.suggest_int("hmm_states", 2, 10)),
                "markov_states": float(trial.suggest_int("markov_states", 2, 6)),
                "drift_gamma_alpha": trial.suggest_float("drift_gamma_alpha", 0.2, 0.95),
                "sigma_floor": trial.suggest_float("sigma_floor", 1e-6, 5e-3, log=True),
                "direction_neutral_threshold": trial.suggest_float(
                    "direction_neutral_threshold", 0.05, 0.6
                ),
                "friction_spread_fraction": trial.suggest_float(
                    "friction_spread_fraction", 0.0, 0.8
                ),
                "friction_per_contract_flat": trial.suggest_float(
                    "friction_per_contract_flat", 0.0, 0.1
                ),
                "mtf_fast_weight": trial.suggest_float("mtf_fast_weight", 0.1, 1.0),
                "mtf_medium_weight": trial.suggest_float("mtf_medium_weight", 0.0, 1.0),
                "mtf_slow_weight": trial.suggest_float("mtf_slow_weight", 0.0, 1.0),
            }
            features = feature_builder(params)
            engine = engine_builder(params)
            _, _, summary = engine.run(
                features,
                option_chain_df,
                mtf_config=MTFWeightConfig(
                    fast_weight=params["mtf_fast_weight"],
                    medium_weight=params["mtf_medium_weight"],
                    slow_weight=params["mtf_slow_weight"],
                ),
            )
            score = float(summary.get(metric, np.nan))
            if not np.isfinite(score):
                score = -1e9
            return score

        study = optuna.create_study(direction=direction)
        study.optimize(
            objective,
            n_trials=max(int(self.config.n_trials), 1),
            timeout=self.config.timeout_seconds,
        )
        best_params = {str(k): float(v) for k, v in study.best_params.items()}
        best_engine = engine_builder(best_params)
        best_features = feature_builder(best_params)
        _, _, best_summary = best_engine.run(
            best_features,
            option_chain_df,
            mtf_config=MTFWeightConfig(
                fast_weight=best_params.get("mtf_fast_weight", 0.6),
                medium_weight=best_params.get("mtf_medium_weight", 0.3),
                slow_weight=best_params.get("mtf_slow_weight", 0.1),
            ),
        )
        return best_params, best_summary, study


class PortfolioBacktestEngine:
    """Runs multiple symbol backtests and combines them with correlation-aware portfolio stats."""

    def run_multi_symbol(
        self,
        *,
        engines: dict[str, BacktestEngine],
        features_by_symbol: dict[str, pd.DataFrame],
        chain_by_symbol: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
        equity_frames: list[pd.DataFrame] = []
        trades_frames: list[pd.DataFrame] = []
        symbol_returns: dict[str, pd.Series] = {}

        for symbol, engine in engines.items():
            if symbol not in features_by_symbol or symbol not in chain_by_symbol:
                continue
            eq, tr, _ = engine.run(features_by_symbol[symbol], chain_by_symbol[symbol])
            if eq.empty:
                continue
            e = eq[["equity"]].copy()
            e.columns = [symbol]
            equity_frames.append(e)
            symbol_returns[symbol] = e[symbol].pct_change().replace([np.inf, -np.inf], np.nan)
            if not tr.empty:
                t = tr.copy()
                t["symbol"] = symbol
                trades_frames.append(t)

        if not equity_frames:
            return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()

        equity_panel = pd.concat(equity_frames, axis=1).sort_index().ffill()
        portfolio_equity = equity_panel.sum(axis=1).to_frame("equity")
        correlation_matrix = pd.DataFrame(symbol_returns).corr()
        trades_all = (
            pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
        )

        from rlm.backtest.metrics import summarize_backtest

        summary = summarize_backtest(portfolio_equity, trades_all)
        if not correlation_matrix.empty:
            tri = correlation_matrix.where(~np.tri(correlation_matrix.shape[0], dtype=bool))
            summary["avg_cross_symbol_corr"] = (
                float(tri.stack().mean()) if not tri.stack().empty else np.nan
            )
        return portfolio_equity, trades_all, summary, correlation_matrix


def build_fill_config_from_friction(
    *, spread_fraction: float, per_contract_flat: float
) -> FillConfig:
    return FillConfig(
        slippage=SlippageConfig(
            spread_fraction=max(float(spread_fraction), 0.0),
            per_contract_flat=max(float(per_contract_flat), 0.0),
            min_slippage=0.0,
        )
    )
