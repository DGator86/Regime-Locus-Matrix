#!/usr/bin/env python3
"""
Random-search optimize ForecastConfig + rolling windows against a full backtest objective.

Runs FactorPipeline once, then evaluates many (forecast + classify + BacktestEngine) draws.
Uses the same data paths as run_backtest.py.

Examples:

    python scripts/optimize_forecast_params.py --symbol SPY --trials 80 --objective composite
    python scripts/optimize_forecast_params.py --symbol SPY --trials 40 --objective calmar --min-trades 10
    python scripts/optimize_forecast_params.py --synthetic --trials 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_option_chain_csv
from rlm.features.factors.pipeline import FactorPipeline
from rlm.optimization.tuning import random_search_forecast_params


def parse_args() -> argparse.Namespace:
    def _state_count(value: str) -> int:
        iv = int(value)
        if iv < 2 or iv > 15:
            raise argparse.ArgumentTypeError("state counts must be in [2, 15]")
        return iv

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--bars", default=None)
    p.add_argument("--chain", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--warmup-days", type=int, default=220)
    p.add_argument("--no-vix", action="store_true")
    p.add_argument("--trials", type=int, default=60, help="Random parameter draws")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--objective",
        choices=("sharpe", "calmar", "composite", "total_return"),
        default="composite",
        help="composite = sharpe + 0.75 * max_drawdown (drawdown negative)",
    )
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--drawdown-penalty", type=float, default=0.75)
    p.add_argument("--use-hmm", action="store_true", help="Slower: refit HMM each trial")
    p.add_argument(
        "--use-markov", action="store_true", help="Refit Markov-switching model each trial"
    )
    p.add_argument("--hmm-states", type=_state_count, default=6)
    p.add_argument("--hmm-iterations", type=int, default=100)
    p.add_argument("--hmm-filter-backend", choices=("auto", "numpy", "numba"), default="auto")
    p.add_argument("--hmm-prefer-gpu", action="store_true")
    p.add_argument("--markov-states", type=_state_count, default=3)
    p.add_argument(
        "--top",
        type=int,
        default=8,
        help="Print and save this many best trials",
    )
    p.add_argument(
        "--export-json",
        type=str,
        default="data/processed/forecast_param_search.json",
        help="Write top results + best params (repo-relative)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.use_hmm and args.use_markov:
        print("Choose only one regime model: --use-hmm or --use-markov", file=sys.stderr)
        return 1
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    chain_rel = args.chain or rel_option_chain_csv(sym)

    end = pd.Timestamp(date.today()).normalize()
    if args.synthetic:
        bars = synthetic_bars_demo(end, periods=max(args.warmup_days, 120))
        chain = synthetic_option_chain_from_bars(bars, underlying=sym)
    else:
        bp = ROOT / bars_rel
        cp = ROOT / chain_rel
        if not bp.is_file():
            print(f"Missing bars: {bp} (use --synthetic)", file=sys.stderr)
            return 1
        if not cp.is_file():
            print(f"Missing chain: {cp}", file=sys.stderr)
            return 1
        bars = pd.read_csv(bp, parse_dates=["timestamp"])
        bars = bars.sort_values("timestamp").set_index("timestamp")
        chain = pd.read_csv(cp, parse_dates=["timestamp", "expiry"])

    bars = prepare_bars_for_factors(bars, chain, underlying=sym, attach_vix=not args.no_vix)
    factors = FactorPipeline().run(bars)

    rng = np.random.default_rng(args.seed)
    results = random_search_forecast_params(
        factors,
        chain,
        underlying_symbol=sym,
        n_trials=max(1, args.trials),
        rng=rng,
        objective=args.objective,  # type: ignore[arg-type]
        min_trades=args.min_trades,
        use_markov=args.use_markov,
        use_hmm=args.use_hmm,
        hmm_states=args.hmm_states,
        hmm_n_iter=args.hmm_iterations,
        hmm_filter_backend=args.hmm_filter_backend,
        hmm_prefer_gpu=args.hmm_prefer_gpu,
        markov_states=args.markov_states,
        drawdown_penalty=args.drawdown_penalty,
    )

    top_n = min(args.top, len(results))
    print(f"Objective={args.objective}  min_trades={args.min_trades}  trials={args.trials}\n")
    for i in range(top_n):
        score, summary, params = results[i]
        print(f"--- rank {i + 1}  score={score:.6g} ---")
        for k, v in params.items():
            print(f"  {k}: {v}")
        for k in (
            "total_return_pct",
            "sharpe",
            "max_drawdown",
            "num_trades",
            "win_rate",
            "profit_factor",
        ):
            if k in summary:
                print(f"  {k}: {summary[k]}")
        print()

    best_score, best_summary, best_params = results[0]
    export_path = ROOT / args.export_json
    export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": args.objective,
        "min_trades": args.min_trades,
        "seed": args.seed,
        "trials": args.trials,
        "symbol": sym,
        "best_score": best_score,
        "best_params": best_params,
        "best_summary": {
            k: float(v) for k, v in best_summary.items() if isinstance(v, (int, float))
        },
        "top": [
            {
                "score": float(r[0]),
                "params": r[2],
                "summary": {k: float(v) for k, v in r[1].items() if isinstance(v, (int, float))},
            }
            for r in results[:top_n]
        ],
    }
    export_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {export_path.relative_to(ROOT)}")
    print(
        "\nSuggested ForecastConfig (Python):\n"
        f"  ForecastConfig(\n"
        f"      drift_gamma_alpha={best_params['drift_gamma_alpha']:.6f},\n"
        f"      sigma_floor={best_params['sigma_floor']:.6e},\n"
        f"      direction_neutral_threshold={best_params['direction_neutral_threshold']:.6f},\n"
        f"  )\n"
        f"  # ForecastPipeline(..., move_window={best_params['move_window']}, "
        f"vol_window={best_params['vol_window']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
