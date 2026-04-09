from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.backtest.engine import (
    BacktestEngine,
    BacktestHyperparameterOptimizer,
    GapRiskStressConfig,
    HyperOptConfig,
    MTFWeightConfig,
    MonteCarloBootstrapConfig,
    PortfolioBacktestEngine,
    build_fill_config_from_friction,
)
from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import (
    DEFAULT_SYMBOL,
    backtest_equity_filename,
    backtest_trades_filename,
    rel_bars_csv,
    rel_option_chain_csv,
)
from rlm.factors.pipeline import FactorPipeline
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
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RLM backtest with optional Optuna tuning, stress tests, and "
            "multi-symbol mode."
        )
    )
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    parser.add_argument(
        "--use-markov", action="store_true", help="Use Markov-switching regime model."
    )
    parser.add_argument("--markov-states", type=int, default=3)
    parser.add_argument(
        "--probabilistic", action="store_true", help="Use probabilistic forecast output."
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Optional quantile model artifact JSON."
    )
    parser.add_argument(
        "--dynamic-sizing", action="store_true", help="Enable Kelly/vol-target sizing."
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly cap to use when dynamic sizing is enabled.",
    )
    parser.add_argument(
        "--no-regime-adjusted-kelly",
        action="store_false",
        dest="regime_adjusted_kelly",
        help="Disable latent-regime Kelly adjustment (enabled by default).",
    )
    parser.add_argument(
        "--vault-uncertainty-threshold",
        type=float,
        default=0.03,
        help="Cut size when forecast 5th-95th range exceeds this return-width threshold.",
    )
    parser.add_argument(
        "--vault-size-multiplier",
        type=float,
        default=0.5,
        help="Position-size multiplier to apply when the Vault rule triggers.",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Restrict the engine to the current calendar date (local).",
    )
    parser.add_argument(
        "--only-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Restrict the engine to this calendar date (after full pipeline warm-up).",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Underlying for default --bars/--chain paths and engine (default {DEFAULT_SYMBOL}).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols for portfolio-level backtest (e.g. SPY,QQQ,IWM).",
    )
    parser.add_argument(
        "--bars",
        type=str,
        default=None,
        help="CSV with timestamp column (default: data/raw/bars_{SYMBOL}.csv).",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Option chain CSV (default: data/raw/option_chain_{SYMBOL}.csv).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Ignore --bars/--chain files and build in-memory demo data (needed if CSVs absent).",
    )
    parser.add_argument(
        "--warmup-days", type=int, default=220, help="Synthetic history length (days)."
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip yfinance VIX/VVIX when enriching bars from the option chain.",
    )
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument("--mtf-fast-weight", type=float, default=0.6)
    parser.add_argument("--mtf-medium-weight", type=float, default=0.3)
    parser.add_argument("--mtf-slow-weight", type=float, default=0.1)
    parser.add_argument("--friction-spread-fraction", type=float, default=0.25)
    parser.add_argument("--friction-per-contract-flat", type=float, default=0.0)
    parser.add_argument("--mc-bootstrap-paths", type=int, default=0)
    parser.add_argument("--mc-sample-frac", type=float, default=1.0)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument(
        "--gap-risk-bps",
        type=str,
        default="0,75,150,300",
        help="Comma-separated downside gap shocks in bps.",
    )
    parser.add_argument(
        "--gap-regime-multipliers",
        type=str,
        default=None,
        help='JSON dict of regime multipliers, e.g. {"bear|high|thin|short_gamma":1.5}',
    )
    return parser.parse_args()


def _load_or_synthetic_bars(
    path: str, *, synthetic: bool, end: pd.Timestamp, warmup_days: int
) -> pd.DataFrame:
    if synthetic:
        return synthetic_bars_demo(end, periods=max(warmup_days, 120))
    p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(f"Bars file not found: {p}. Pass --synthetic or create the CSV.")
    bars = pd.read_csv(p, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")
    return bars


def _load_or_synthetic_chain(
    path: str, bars: pd.DataFrame, *, synthetic: bool, underlying: str
) -> pd.DataFrame:
    if synthetic:
        return synthetic_option_chain_from_bars(bars, underlying=underlying)
    p = ROOT / path
    if not p.is_file():
        raise FileNotFoundError(f"Chain file not found: {p}. Pass --synthetic or create the CSV.")
    return pd.read_csv(p, parse_dates=["timestamp", "expiry"])


def _build_features(
    *,
    bars: pd.DataFrame,
    chain: pd.DataFrame,
    args: argparse.Namespace,
    params: dict[str, float] | None = None,
) -> pd.DataFrame:
    tuned = params or {}
    fc = ForecastConfig(
        drift_gamma_alpha=float(tuned.get("drift_gamma_alpha", 0.65)),
        sigma_floor=float(tuned.get("sigma_floor", 1e-4)),
        direction_neutral_threshold=float(tuned.get("direction_neutral_threshold", 0.3)),
    )
    hmm_states = int(round(tuned.get("hmm_states", args.hmm_states)))
    markov_states = int(round(tuned.get("markov_states", args.markov_states)))

    features = FactorPipeline().run(
        prepare_bars_for_factors(bars, chain, underlying=args.symbol, attach_vix=not args.no_vix)
    )

    if args.use_hmm and args.probabilistic:
        return HybridProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=hmm_states),
            model_path=args.model_path,
        ).run(features)
    if args.use_hmm:
        return HybridForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=hmm_states),
        ).run(features)
    if args.use_markov and args.probabilistic:
        return HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=markov_states),
            model_path=args.model_path,
        ).run(features)
    if args.use_markov:
        return HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=markov_states),
        ).run(features)
    if args.probabilistic:
        return ProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            model_path=args.model_path,
        ).run(features)
    return ForecastPipeline(config=fc, move_window=100, vol_window=100).run(features)


def main() -> None:
    args = parse_args()
    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")

    sym = str(args.symbol).upper().strip()
    if args.today and args.only_date:
        raise SystemExit("Use either --today or --only-date, not both.")

    if args.today:
        only_ts = pd.Timestamp(date.today())
    elif args.only_date:
        only_ts = pd.Timestamp(datetime.strptime(args.only_date, "%Y-%m-%d").date())
    else:
        only_ts = None

    symbols = [s.strip().upper() for s in (args.symbols or sym).split(",") if s.strip()]
    if not symbols:
        symbols = [sym]

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    chain_by_symbol: dict[str, pd.DataFrame] = {}
    features_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        bars_rel = rel_bars_csv(symbol) if args.bars is None or len(symbols) > 1 else args.bars
        chain_rel = (
            rel_option_chain_csv(symbol) if args.chain is None or len(symbols) > 1 else args.chain
        )
        end = only_ts if only_ts is not None else pd.Timestamp.today().normalize()
        bars = _load_or_synthetic_bars(
            bars_rel,
            synthetic=args.synthetic,
            end=end,
            warmup_days=args.warmup_days,
        )
        chain = _load_or_synthetic_chain(
            chain_rel, bars, synthetic=args.synthetic, underlying=symbol
        )
        bars_by_symbol[symbol] = bars
        chain_by_symbol[symbol] = chain
        args.symbol = symbol
        features_by_symbol[symbol] = _build_features(bars=bars, chain=chain, args=args)

    base_mtf = MTFWeightConfig(
        fast_weight=args.mtf_fast_weight,
        medium_weight=args.mtf_medium_weight,
        slow_weight=args.mtf_slow_weight,
    )

    gap_levels = tuple(float(x.strip()) for x in args.gap_risk_bps.split(",") if x.strip())
    gap_mult = json.loads(args.gap_regime_multipliers) if args.gap_regime_multipliers else None
    gap_cfg = GapRiskStressConfig(downside_gap_bps=gap_levels, regime_multipliers=gap_mult)
    mc_cfg = None
    if args.mc_bootstrap_paths > 0:
        mc_cfg = MonteCarloBootstrapConfig(
            n_paths=args.mc_bootstrap_paths,
            sample_frac=args.mc_sample_frac,
            random_seed=args.mc_seed,
        )

    def make_engine(params: dict[str, float] | None = None, *, symbol: str = sym) -> BacktestEngine:
        tuned = params or {}
        fill_cfg = build_fill_config_from_friction(
            spread_fraction=float(
                tuned.get("friction_spread_fraction", args.friction_spread_fraction)
            ),
            per_contract_flat=float(
                tuned.get("friction_per_contract_flat", args.friction_per_contract_flat)
            ),
        )
        return BacktestEngine(
            initial_capital=100_000.0,
            contract_multiplier=100,
            strike_increment=5.0,
            underlying_symbol=symbol,
            quantity_per_trade=1,
            fill_config=fill_cfg,
            roee_config=ROEEConfig(
                use_dynamic_sizing=args.dynamic_sizing,
                max_kelly_fraction=args.kelly_fraction,
                regime_adjusted_kelly=args.regime_adjusted_kelly,
                vault_uncertainty_threshold=args.vault_uncertainty_threshold,
                vault_size_multiplier=args.vault_size_multiplier,
            ),
        )

    if len(symbols) == 1:
        symbol = symbols[0]
        option_chain = chain_by_symbol[symbol]

        tuned_params: dict[str, float] | None = None
        if args.optuna_trials > 0:
            optimizer = BacktestHyperparameterOptimizer(
                HyperOptConfig(
                    n_trials=args.optuna_trials,
                    timeout_seconds=args.optuna_timeout,
                    metric="sharpe",
                )
            )
            tuned_params, best_summary, _ = optimizer.optimize(
                feature_builder=lambda p: _build_features(
                    bars=bars_by_symbol[symbol],
                    chain=chain_by_symbol[symbol],
                    args=args,
                    params=p,
                ),
                option_chain_df=option_chain,
                engine_builder=lambda p: make_engine(p, symbol=symbol),
            )
            print("Optuna best params:")
            print(tuned_params)
            print("Optuna best summary:")
            print(best_summary)

        engine = make_engine(tuned_params, symbol=symbol)
        features_run = _build_features(
            bars=bars_by_symbol[symbol],
            chain=chain_by_symbol[symbol],
            args=args,
            params=tuned_params,
        )

        if only_ts is not None:
            d = only_ts.normalize()
            feat_mask = features_run.index.normalize() == d
            features_run = features_run.loc[feat_mask]
            option_chain = option_chain.loc[
                pd.to_datetime(option_chain["timestamp"]).dt.normalize() == d
            ].copy()
            if features_run.empty:
                raise SystemExit(f"No bar rows on {d.date()} after warm-up. Check data range.")
            if option_chain.empty:
                raise SystemExit(f"No option chain rows on {d.date()}. Check chain coverage.")

        tuned_mtf = base_mtf
        if tuned_params is not None:
            tuned_mtf = MTFWeightConfig(
                fast_weight=float(tuned_params.get("mtf_fast_weight", base_mtf.fast_weight)),
                medium_weight=float(tuned_params.get("mtf_medium_weight", base_mtf.medium_weight)),
                slow_weight=float(tuned_params.get("mtf_slow_weight", base_mtf.slow_weight)),
            )

        equity_frame, trades_frame, summary, diagnostics = engine.run_with_robustness(
            features_run,
            option_chain,
            mtf_config=tuned_mtf,
            monte_carlo=mc_cfg,
            gap_risk=gap_cfg,
        )

        print("Backtest summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if diagnostics:
            print("Robustness diagnostics:")
            for section, payload in diagnostics.items():
                print(f"  {section}: {payload}")

        out_dir = ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        equity_path = out_dir / backtest_equity_filename(symbol)
        trades_path = out_dir / backtest_trades_filename(symbol)
        equity_frame.to_csv(equity_path)
        trades_frame.to_csv(trades_path, index=False)
        print(f"Wrote {equity_path.relative_to(ROOT)} and {trades_path.relative_to(ROOT)}.")
        return

    engines = {symbol: make_engine(symbol=symbol) for symbol in symbols}
    portfolio_runner = PortfolioBacktestEngine()
    equity_frame, trades_frame, summary, corr = portfolio_runner.run_multi_symbol(
        engines=engines,
        features_by_symbol=features_by_symbol,
        chain_by_symbol=chain_by_symbol,
    )
    print("Portfolio summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if not corr.empty:
        print("Cross-symbol correlation matrix:")
        print(corr)


if __name__ == "__main__":
    main()
