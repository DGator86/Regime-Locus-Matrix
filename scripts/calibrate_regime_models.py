#!/usr/bin/env python3
"""
Weekly champion/challenger calibration for regime models.

Runs the same forecast parameter samples through both HMM and Markov overlays,
then promotes the higher-Sharpe contender to the live model config consumed by
the live trading scripts.

Examples:

    python scripts/calibrate_regime_models.py --symbol SPY --trials 24 --no-vix
    python scripts/calibrate_regime_models.py --synthetic --trials 6 --lookback-bars 220
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_option_chain_csv
from rlm.factors.pipeline import FactorPipeline
from rlm.factors.multi_timeframe import MultiTimeframeEngine, format_precompute_instructions, parse_higher_tfs
from rlm.forecasting.live_model import (
    LiveForecastParameters,
    LiveHMMParameters,
    LiveMarkovParameters,
    LiveRegimeModelConfig,
    save_live_regime_model,
)
from rlm.optimization.tuning import generate_forecast_param_samples, random_search_forecast_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--bars", default=None)
    p.add_argument("--chain", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--warmup-days", type=int, default=260)
    p.add_argument("--lookback-bars", type=int, default=520, help="Only calibrate on the most recent N bars")
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--trials", type=int, default=24, help="Shared parameter samples per contender")
    p.add_argument("--mtf", action="store_true", help="Enable multi-timeframe factor augmentation.")
    p.add_argument(
        "--higher-tfs",
        type=str,
        default="1W,1M",
        help="Comma-separated higher-timeframe resample rules for --mtf (example: 1W,1M).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--objective",
        choices=("sharpe", "calmar", "composite", "total_return"),
        default="sharpe",
        help="Search objective within each contender. Promotion still keys off Sharpe first.",
    )
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--drawdown-penalty", type=float, default=0.75)
    p.add_argument("--top", type=int, default=5, help="Persist this many top runs per contender")
    p.add_argument("--hmm-states", type=int, default=6)
    p.add_argument("--hmm-iterations", type=int, default=100)
    p.add_argument(
        "--hmm-filter-backend",
        choices=("auto", "numpy", "numba"),
        default="auto",
        help="Forward-filter backend for HMM inference.",
    )
    p.add_argument(
        "--hmm-prefer-gpu",
        action="store_true",
        help="Record a GPU preference for the HMM runtime when CUDA is available.",
    )
    p.add_argument("--markov-states", type=int, default=3)
    p.add_argument(
        "--export-json",
        type=str,
        default="data/processed/regime_model_calibration.json",
        help="Calibration report path (repo-relative)",
    )
    p.add_argument(
        "--promote-path",
        type=str,
        default="data/processed/live_regime_model.json",
        help="Promoted live-model config path (repo-relative)",
    )
    p.add_argument("--no-promote", action="store_true", help="Write the report but do not update live model")
    return p.parse_args()


def _finite_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, float("-inf"))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return out if np.isfinite(out) else float("-inf")


def _top_payload(
    results: list[tuple[float, dict[str, float], dict[str, Any]]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for score, summary, params in results[:top_n]:
        payload.append(
            {
                "score": float(score),
                "summary": {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))},
                "params": params,
            }
        )
    return payload


def _build_live_config(
    *,
    champion_model: str,
    champion_params: dict[str, Any],
    args: argparse.Namespace,
    report_path: Path,
) -> LiveRegimeModelConfig:
    return LiveRegimeModelConfig(
        model=champion_model,
        forecast=LiveForecastParameters(
            drift_gamma_alpha=float(champion_params["drift_gamma_alpha"]),
            sigma_floor=float(champion_params["sigma_floor"]),
            direction_neutral_threshold=float(champion_params["direction_neutral_threshold"]),
            move_window=int(champion_params["move_window"]),
            vol_window=int(champion_params["vol_window"]),
        ),
        hmm=LiveHMMParameters(
            n_states=int(args.hmm_states),
            n_iter=int(args.hmm_iterations),
            filter_backend=str(args.hmm_filter_backend),
            prefer_gpu=bool(args.hmm_prefer_gpu),
        ),
        markov=LiveMarkovParameters(
            n_states=int(args.markov_states),
        ),
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_report": str(report_path.relative_to(ROOT)),
            "selection_metric": "sharpe",
            "search_objective": str(args.objective),
            "trials": int(args.trials),
            "seed": int(args.seed),
            "symbol": str(args.symbol).upper().strip(),
        },
    )


def main() -> int:
    args = parse_args()
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    chain_rel = args.chain or rel_option_chain_csv(sym)

    end = pd.Timestamp(date.today()).normalize()
    if args.synthetic:
        bars = synthetic_bars_demo(end, periods=max(args.warmup_days, args.lookback_bars or 0, 160))
        chain = synthetic_option_chain_from_bars(bars, underlying=sym)
    else:
        bars_path = ROOT / bars_rel
        chain_path = ROOT / chain_rel
        if not bars_path.is_file():
            print(f"Missing bars: {bars_path} (use --synthetic)", file=sys.stderr)
            return 1
        if not chain_path.is_file():
            print(f"Missing chain: {chain_path}", file=sys.stderr)
            return 1
        bars = pd.read_csv(bars_path, parse_dates=["timestamp"])
        bars = bars.sort_values("timestamp").set_index("timestamp")
        chain = pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"])

    if args.lookback_bars and int(args.lookback_bars) > 0:
        bars = bars.iloc[-int(args.lookback_bars) :].copy()
        chain = chain[chain["timestamp"].isin(bars.index)].copy()

    bars = prepare_bars_for_factors(bars, chain, underlying=sym, attach_vix=not args.no_vix)
    factors = FactorPipeline().run(bars)
    if args.mtf:
        higher_tfs = parse_higher_tfs(args.higher_tfs)
        factors = MultiTimeframeEngine(higher_tfs=higher_tfs).augment_factors(bars, factors)
        print(format_precompute_instructions(symbol=sym, higher_tfs=higher_tfs))

    samples = generate_forecast_param_samples(
        n_trials=max(1, int(args.trials)),
        rng=np.random.default_rng(args.seed),
    )

    hmm_results = random_search_forecast_params(
        factors,
        chain,
        underlying_symbol=sym,
        n_trials=len(samples),
        rng=np.random.default_rng(args.seed),
        objective=args.objective,  # type: ignore[arg-type]
        min_trades=args.min_trades,
        regime_model="hmm",
        hmm_states=args.hmm_states,
        hmm_n_iter=args.hmm_iterations,
        hmm_filter_backend=args.hmm_filter_backend,
        hmm_prefer_gpu=args.hmm_prefer_gpu,
        drawdown_penalty=args.drawdown_penalty,
        samples=samples,
    )
    markov_results = random_search_forecast_params(
        factors,
        chain,
        underlying_symbol=sym,
        n_trials=len(samples),
        rng=np.random.default_rng(args.seed + 1),
        objective=args.objective,  # type: ignore[arg-type]
        min_trades=args.min_trades,
        regime_model="markov",
        markov_states=args.markov_states,
        drawdown_penalty=args.drawdown_penalty,
        samples=samples,
    )

    hmm_best_score, hmm_best_summary, hmm_best_params = hmm_results[0]
    markov_best_score, markov_best_summary, markov_best_params = markov_results[0]
    hmm_sharpe = _finite_metric(hmm_best_summary, "sharpe")
    markov_sharpe = _finite_metric(markov_best_summary, "sharpe")

    champion_model = "hmm"
    champion_score = float(hmm_best_score)
    champion_summary = hmm_best_summary
    champion_params = hmm_best_params
    challenger_model = "markov"
    challenger_score = float(markov_best_score)
    challenger_summary = markov_best_summary
    challenger_params = markov_best_params
    if (markov_sharpe, float(markov_best_score)) > (hmm_sharpe, float(hmm_best_score)):
        champion_model = "markov"
        champion_score = float(markov_best_score)
        champion_summary = markov_best_summary
        champion_params = markov_best_params
        challenger_model = "hmm"
        challenger_score = float(hmm_best_score)
        challenger_summary = hmm_best_summary
        challenger_params = hmm_best_params

    export_path = ROOT / args.export_json
    export_path.parent.mkdir(parents=True, exist_ok=True)
    promote_path = ROOT / args.promote_path

    live_config = _build_live_config(
        champion_model=champion_model,
        champion_params=champion_params,
        args=args,
        report_path=export_path,
    )
    if not args.no_promote:
        save_live_regime_model(live_config, promote_path)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "objective": args.objective,
        "selection_metric": "sharpe",
        "seed": args.seed,
        "trials": len(samples),
        "lookback_bars": args.lookback_bars,
        "promotion_performed": not args.no_promote,
        "promote_path": str(promote_path.relative_to(ROOT)),
        "champion": {
            "model": champion_model,
            "score": champion_score,
            "sharpe": _finite_metric(champion_summary, "sharpe"),
            "summary": {k: float(v) for k, v in champion_summary.items() if isinstance(v, (int, float))},
            "params": champion_params,
        },
        "challenger": {
            "model": challenger_model,
            "score": challenger_score,
            "sharpe": _finite_metric(challenger_summary, "sharpe"),
            "summary": {k: float(v) for k, v in challenger_summary.items() if isinstance(v, (int, float))},
            "params": challenger_params,
        },
        "contenders": {
            "hmm": {
                "best_score": float(hmm_best_score),
                "best_summary": {k: float(v) for k, v in hmm_best_summary.items() if isinstance(v, (int, float))},
                "best_params": hmm_best_params,
                "top": _top_payload(hmm_results, top_n=min(args.top, len(hmm_results))),
            },
            "markov": {
                "best_score": float(markov_best_score),
                "best_summary": {k: float(v) for k, v in markov_best_summary.items() if isinstance(v, (int, float))},
                "best_params": markov_best_params,
                "top": _top_payload(markov_results, top_n=min(args.top, len(markov_results))),
            },
        },
    }
    export_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Champion: {champion_model.upper()}  sharpe={_finite_metric(champion_summary, 'sharpe'):.6g}")
    print(f"Challenger: {challenger_model.upper()}  sharpe={_finite_metric(challenger_summary, 'sharpe'):.6g}")
    print(f"Wrote calibration report: {export_path.relative_to(ROOT)}")
    if args.no_promote:
        print("Live promotion skipped (--no-promote).")
    else:
        print(f"Promoted live model: {promote_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
