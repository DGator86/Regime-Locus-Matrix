from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import (
    DEFAULT_SYMBOL,
    rel_bars_csv,
    rel_forecast_features_csv,
    rel_option_chain_csv,
)
from rlm.factors.pipeline import FactorPipeline
from rlm.factors.multi_timeframe import MultiTimeframeEngine, format_precompute_instructions, parse_higher_tfs
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.pipeline import (
    ForecastPipeline,
    HybridForecastPipeline,
    HybridMarkovForecastPipeline,
    HybridProbabilisticForecastPipeline,
)
from rlm.forecasting.probabilistic import ProbabilisticForecastPipeline
from rlm.types.forecast import ForecastConfig


def _parse_tf_key_values(text: str | None, cast_float: bool = False) -> dict[str, str | float]:
    if not text:
        return {}
    out: dict[str, str | float] = {}
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"Invalid key=value pair: {part}")
        key, val = part.split("=", 1)
        key = key.strip()
        if cast_float:
            out[key] = float(val.strip())
        else:
            out[key] = val.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factors → forecast features (default bars: data/raw/bars_{SYMBOL}.csv)."
    )
    parser.add_argument("--use-hmm", action="store_true")
    parser.add_argument("--hmm-states", type=int, default=6)
    parser.add_argument(
        "--use-markov", action="store_true", help="Use Markov-switching regime model."
    )
    parser.add_argument("--markov-states", type=int, default=3)
    parser.add_argument(
        "--mtf-regimes", action="store_true", help="Blend LTF regime probs with HTF parquet probs."
    )
    parser.add_argument(
        "--mtf-htf-prob-paths",
        default=None,
        help=(
            "Comma-separated timeframe=parquet_path entries, e.g. "
            "1H=data/processed/htf_1h.parquet,1D=data/processed/htf_1d.parquet"
        ),
    )
    parser.add_argument(
        "--mtf-htf-weights",
        default=None,
        help="Comma-separated timeframe=weight entries, e.g. 1H=0.4,1D=0.6",
    )
    parser.add_argument(
        "--mtf-ltf-weight", type=float, default=0.7, help="Blend weight on LTF model probabilities."
    )
    parser.add_argument(
        "--probabilistic", action="store_true", help="Use probabilistic forecast output."
    )
    parser.add_argument("--model-path", default=None, help="Optional quantile model artifact JSON.")
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Ticker for default paths (default {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--bars",
        default=None,
        help="Bars CSV relative to repo root (default: data/raw/bars_{SYMBOL}.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV relative to repo root "
            "(default: data/processed/forecast_features_{SYMBOL}.csv)"
        ),
    )
    parser.add_argument(
        "--chain",
        default=None,
        help=(
            "Option chain CSV for enrichment "
            "(default: data/raw/option_chain_{SYMBOL}.csv if present)"
        ),
    )
    parser.add_argument("--no-vix", action="store_true", help="Skip yfinance VIX/VVIX.")
    parser.add_argument("--mtf", action="store_true", help="Enable multi-timeframe factor augmentation.")
    parser.add_argument(
        "--higher-tfs",
        default="1W,1M",
        help="Comma-separated higher-timeframe resample rules for --mtf (example: 1W,1M).",
    )
    parser.add_argument(
        "--no-kronos",
        action="store_true",
        help="Disable Kronos regime-confidence and forecast overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_hmm and args.use_markov:
        raise SystemExit("Use either --use-hmm or --use-markov, not both.")
    if args.mtf_regimes and args.use_markov:
        raise SystemExit(
            "--mtf-regimes currently supports only the HMM LTF model in this pipeline."
        )
    if args.mtf_regimes and not args.use_hmm:
        raise SystemExit("--mtf-regimes requires --use-hmm.")
    mtf_paths = {k: str(v) for k, v in _parse_tf_key_values(args.mtf_htf_prob_paths).items()}
    mtf_weights = {
        k: float(v) for k, v in _parse_tf_key_values(args.mtf_htf_weights, cast_float=True).items()
    }
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    out_rel = args.out or rel_forecast_features_csv(sym)
    bars_path = ROOT / bars_rel
    if not bars_path.is_file():
        raise SystemExit(
            f"Bars file not found: {bars_path}\n"
            "See: python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol "
            f"{sym} --start YYYY-MM-DD"
        )

    df = pd.read_csv(bars_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    chain_rel = args.chain or rel_option_chain_csv(sym)
    cp = ROOT / chain_rel
    opch = pd.read_csv(cp, parse_dates=["timestamp", "expiry"]) if cp.is_file() else None
    df = prepare_bars_for_factors(df, opch, underlying=sym, attach_vix=not args.no_vix)

    factor_pipeline = FactorPipeline()

    factors = factor_pipeline.run(df)
    if args.mtf:
        higher_tfs = parse_higher_tfs(args.higher_tfs)
        factors = MultiTimeframeEngine(higher_tfs=higher_tfs).augment_factors(df, factors)
        print(format_precompute_instructions(symbol=sym, higher_tfs=higher_tfs))
    fc = ForecastConfig(
        drift_gamma_alpha=0.65,
        sigma_floor=1e-4,
        direction_neutral_threshold=0.3,
    )
    if args.use_hmm and args.probabilistic:
        forecast = HybridProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
            model_path=args.model_path,
        ).run(factors)
    elif args.use_hmm:
        forecast = HybridForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            hmm_config=HMMConfig(n_states=args.hmm_states),
            mtf_regimes=args.mtf_regimes,
            mtf_htf_prob_paths=mtf_paths,
            mtf_htf_weights=mtf_weights,
            mtf_ltf_weight=args.mtf_ltf_weight,
        ).run(factors)
    elif args.use_markov and args.probabilistic:
        forecast = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
            model_path=args.model_path,
        ).run(factors)
    elif args.use_markov:
        forecast = HybridMarkovForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            markov_config=MarkovSwitchingConfig(n_states=args.markov_states),
        ).run(factors)
    elif args.probabilistic:
        forecast = ProbabilisticForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
            model_path=args.model_path,
        ).run(factors)
    else:
        forecast = ForecastPipeline(
            config=fc,
            move_window=100,
            vol_window=100,
        ).run(factors)

    if not args.no_kronos:
        from rlm.kronos import KronosRegimeConfidence

        print("Running Kronos regime-confidence overlay ...")
        forecast = KronosRegimeConfidence().annotate(forecast)

    out_cols = [
        "close",
        "S_D",
        "S_V",
        "S_L",
        "S_G",
        "b_m",
        "b_sigma",
        "mu",
        "sigma",
        "mean_price",
        "lower_1s",
        "upper_1s",
        "lower_2s",
        "upper_2s",
        "forecast_return_lower",
        "forecast_return_median",
        "forecast_return_upper",
        "forecast_uncertainty",
        "realized_vol",
        "forecast_source",
    ]
    if not args.no_kronos:
        out_cols.extend([
            "kronos_confidence",
            "kronos_regime_agreement",
            "kronos_predicted_regime",
            "kronos_transition_flag",
            "kronos_forecast_return",
            "kronos_forecast_vol",
        ])
    if args.use_hmm:
        out_cols.extend(["hmm_state", "hmm_state_label"])
    if args.mtf_regimes:
        out_cols.extend(["mtf_state", "mtf_state_label", "mtf_confidence"])
    if args.use_markov:
        out_cols.extend(["markov_state", "markov_state_label"])

    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out_path)
    print(forecast[out_cols].tail(10))
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
