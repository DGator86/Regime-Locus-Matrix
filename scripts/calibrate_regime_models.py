#!/usr/bin/env python3
"""
Weekly champion/challenger calibration for regime models.

Runs shared forecast-parameter samples through HMM and Markov overlays across
multiple regime-granularity candidates, then promotes the best Sharpe contender
into ``live_regime_model.json``.
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

from rlm.factors.multi_timeframe import (
    MultiTimeframeEngine,
    format_precompute_instructions,
    parse_higher_tfs,
)
from rlm.factors.pipeline import FactorPipeline
from rlm.optimization.tuning import (
    ForecastParamSample,
    generate_forecast_param_samples,
    random_search_forecast_params,
)

from rlm.datasets.backtest_data import synthetic_bars_demo, synthetic_option_chain_from_bars
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_option_chain_csv
from rlm.forecasting.engines import ForecastPipeline
from rlm.forecasting.hmm import RLMHMM, HMMConfig
from rlm.forecasting.live_model import (
    LiveForecastParameters,
    LiveHMMParameters,
    LiveMarkovParameters,
    LiveRegimeModelConfig,
    save_live_regime_model,
)
from rlm.forecasting.markov_switching import MarkovSwitchingConfig, RLMMarkovSwitching


def _bounded_state(value: str) -> int:
    iv = int(value)
    if iv < 2 or iv > 15:
        raise argparse.ArgumentTypeError("state counts must be in [2, 15]")
    return iv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--bars", default=None)
    p.add_argument("--chain", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--warmup-days", type=int, default=260)
    p.add_argument(
        "--lookback-bars", type=int, default=520, help="Only calibrate on the most recent N bars"
    )
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
    p.add_argument("--hmm-states", type=_bounded_state, default=6)
    p.add_argument("--markov-states", type=_bounded_state, default=3)
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
    p.add_argument("--hierarchical", action="store_true", default=True)
    p.add_argument("--macro-weight", type=float, default=0.45)
    p.add_argument(
        "--micro-timeframes",
        nargs="+",
        default=["5min", "1min"],
        help="Micro regime bars used for metadata and live-model config.",
    )
    p.add_argument(
        "--mtf-regimes",
        action="store_true",
        help="Enable multi-timeframe regime blending metadata for downstream forecast runs.",
    )
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
    p.add_argument(
        "--no-promote", action="store_true", help="Write the report but do not update live model"
    )
    return p.parse_args()


def _finite_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, float("-inf"))
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return out if np.isfinite(out) else float("-inf")


def _summary_to_json(summary: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}


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
                "summary": _summary_to_json(summary),
                "params": params,
            }
        )
    return payload


def _resolve_state_sweeps(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    hmm_candidates = sorted({int(args.hmm_states), 8, 10, 12, 15})
    markov_candidates = sorted({int(args.markov_states), 4, 5, 7})
    return [s for s in hmm_candidates if 2 <= s <= 15], [
        s for s in markov_candidates if 2 <= s <= 15
    ]


def _compute_model_criteria(
    *,
    factors: pd.DataFrame,
    sample: ForecastParamSample,
    model_name: str,
    states: int,
    hmm_n_iter: int,
    seed: int,
) -> dict[str, float | None]:
    framed = ForecastPipeline(
        move_window=sample.move_window,
        vol_window=sample.vol_window,
        config=LiveForecastParameters(
            drift_gamma_alpha=sample.drift_gamma_alpha,
            sigma_floor=sample.sigma_floor,
            direction_neutral_threshold=sample.direction_neutral_threshold,
        ).to_forecast_config(),
    ).run(factors)

    if model_name == "hmm":
        model = RLMHMM(
            HMMConfig(
                n_states=states,
                n_iter=hmm_n_iter,
                random_state=seed,
            )
        ).fit(framed, verbose=False)
        if model.model is None:
            return {"loglik": None, "aic": None, "bic": None}
        obs = model.prepare_observations(framed)
        return {
            "loglik": float(model.model.score(obs)),
            "aic": float(model.model.aic(obs)),
            "bic": float(model.model.bic(obs)),
        }

    model = RLMMarkovSwitching(
        MarkovSwitchingConfig(
            n_states=states,
        )
    ).fit(framed, verbose=False)
    if model.fit_result is None:
        return {"loglik": None, "aic": None, "bic": None}
    return {
        "loglik": float(model.fit_result.llf),
        "aic": float(model.fit_result.aic),
        "bic": float(model.fit_result.bic),
    }


def _run_state_tournament(
    *,
    model_name: str,
    states: list[int],
    args: argparse.Namespace,
    factors: pd.DataFrame,
    chain: pd.DataFrame,
    samples: list[ForecastParamSample],
    sym: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_state: dict[str, Any] = {}
    best_row: dict[str, Any] | None = None

    for i, state_count in enumerate(states):
        run_seed = int(args.seed + (1000 * (i + 1)) + (0 if model_name == "hmm" else 500_000))
        run = random_search_forecast_params(
            factors,
            chain,
            underlying_symbol=sym,
            n_trials=len(samples),
            rng=np.random.default_rng(run_seed),
            objective=args.objective,  # type: ignore[arg-type]
            min_trades=args.min_trades,
            regime_model=model_name,  # type: ignore[arg-type]
            hmm_states=state_count,
            hmm_n_iter=args.hmm_iterations,
            hmm_filter_backend=args.hmm_filter_backend,
            hmm_prefer_gpu=args.hmm_prefer_gpu,
            markov_states=state_count,
            drawdown_penalty=args.drawdown_penalty,
            samples=samples,
        )
        best_score, best_summary, best_params = run[0]
        criteria = _compute_model_criteria(
            factors=factors,
            sample=samples[0],
            model_name=model_name,
            states=state_count,
            hmm_n_iter=args.hmm_iterations,
            seed=run_seed,
        )
        state_payload = {
            "states": state_count,
            "best_score": float(best_score),
            "best_sharpe": _finite_metric(best_summary, "sharpe"),
            "best_summary": _summary_to_json(best_summary),
            "best_params": best_params,
            "top": _top_payload(run, top_n=min(args.top, len(run))),
            "criteria": criteria,
        }
        by_state[str(state_count)] = state_payload

        candidate_cmp = (_finite_metric(best_summary, "sharpe"), float(best_score))
        if best_row is None:
            best_row = state_payload
        else:
            incumbent_cmp = (float(best_row["best_sharpe"]), float(best_row["best_score"]))
            if candidate_cmp > incumbent_cmp:
                best_row = state_payload

    if best_row is None:
        raise RuntimeError("No tournament rows produced")

    return by_state, best_row


def _build_live_config(
    *,
    champion_model: str,
    champion_params: dict[str, Any],
    champion_state_count: int,
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
            n_states=int(champion_state_count if champion_model == "hmm" else args.hmm_states),
            n_iter=int(args.hmm_iterations),
            filter_backend=str(args.hmm_filter_backend),
            prefer_gpu=bool(args.hmm_prefer_gpu),
            hierarchical=bool(args.hierarchical),
            macro_weight=float(args.macro_weight),
            micro_timeframes=tuple(str(x) for x in args.micro_timeframes),
        ),
        markov=LiveMarkovParameters(
            n_states=int(
                champion_state_count if champion_model == "markov" else args.markov_states
            ),
            hierarchical=bool(args.hierarchical),
            macro_weight=float(args.macro_weight),
            micro_timeframes=tuple(str(x) for x in args.micro_timeframes),
        ),
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_report": str(report_path.relative_to(ROOT)),
            "selection_metric": "sharpe",
            "search_objective": str(args.objective),
            "trials": int(args.trials),
            "seed": int(args.seed),
            "symbol": str(args.symbol).upper().strip(),
            "mtf_regimes": bool(args.mtf_regimes),
            "hierarchical": bool(args.hierarchical),
            "macro_weight": float(args.macro_weight),
            "micro_timeframes": [str(x) for x in args.micro_timeframes],
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

    hmm_states, markov_states = _resolve_state_sweeps(args)
    hmm_by_state, hmm_best = _run_state_tournament(
        model_name="hmm",
        states=hmm_states,
        args=args,
        factors=factors,
        chain=chain,
        samples=samples,
        sym=sym,
    )
    markov_by_state, markov_best = _run_state_tournament(
        model_name="markov",
        states=markov_states,
        args=args,
        factors=factors,
        chain=chain,
        samples=samples,
        sym=sym,
    )

    hmm_cmp = (float(hmm_best["best_sharpe"]), float(hmm_best["best_score"]))
    markov_cmp = (float(markov_best["best_sharpe"]), float(markov_best["best_score"]))

    champion_model = "hmm"
    champion = hmm_best
    challenger_model = "markov"
    challenger = markov_best
    if markov_cmp > hmm_cmp:
        champion_model = "markov"
        champion = markov_best
        challenger_model = "hmm"
        challenger = hmm_best

    export_path = ROOT / args.export_json
    export_path.parent.mkdir(parents=True, exist_ok=True)
    promote_path = ROOT / args.promote_path

    live_config = _build_live_config(
        champion_model=champion_model,
        champion_params=champion["best_params"],
        champion_state_count=int(champion["states"]),
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
        "mtf_regimes": bool(args.mtf_regimes),
        "promote_path": str(promote_path.relative_to(ROOT)),
        "hierarchical": {
            "enabled": bool(args.hierarchical),
            "macro_weight": float(args.macro_weight),
            "micro_timeframes": [str(x) for x in args.micro_timeframes],
        },
        "sweeps": {
            "hmm_states": hmm_states,
            "markov_states": markov_states,
        },
        "champion": {
            "model": champion_model,
            "score": champion_score,
            "sharpe": _finite_metric(champion_summary, "sharpe"),
            "summary": {
                k: float(v) for k, v in champion_summary.items() if isinstance(v, (int, float))
            },
            "params": champion_params,
        },
        "challenger": {
            "model": challenger_model,
            "score": challenger_score,
            "sharpe": _finite_metric(challenger_summary, "sharpe"),
            "summary": {
                k: float(v) for k, v in challenger_summary.items() if isinstance(v, (int, float))
            },
            "params": challenger_params,
        },
        "contenders": {
            "hmm": {
                "best_score": float(hmm_best_score),
                "best_summary": {
                    k: float(v) for k, v in hmm_best_summary.items() if isinstance(v, (int, float))
                },
                "best_params": hmm_best_params,
                "top": _top_payload(hmm_results, top_n=min(args.top, len(hmm_results))),
            },
            "markov": {
                "best_score": float(markov_best_score),
                "best_summary": {
                    k: float(v)
                    for k, v in markov_best_summary.items()
                    if isinstance(v, (int, float))
                },
                "best_params": markov_best_params,
                "top": _top_payload(markov_results, top_n=min(args.top, len(markov_results))),
            },
            "states": int(champion["states"]),
            "score": float(champion["best_score"]),
            "sharpe": float(champion["best_sharpe"]),
            "summary": champion["best_summary"],
            "params": champion["best_params"],
            "criteria": champion["criteria"],
        },
        "challenger": {
            "model": challenger_model,
            "states": int(challenger["states"]),
            "score": float(challenger["best_score"]),
            "sharpe": float(challenger["best_sharpe"]),
            "summary": challenger["best_summary"],
            "params": challenger["best_params"],
            "criteria": challenger["criteria"],
        },
        "contenders": {
            "hmm": {"by_state": hmm_by_state, "best": hmm_best},
            "markov": {"by_state": markov_by_state, "best": markov_best},
        },
    }
    export_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    champion_sharpe = _finite_metric(champion_summary, "sharpe")
    challenger_sharpe = _finite_metric(challenger_summary, "sharpe")
    print(f"Champion: {champion_model.upper()}  sharpe={champion_sharpe:.6g}")
    print(f"Challenger: {challenger_model.upper()}  sharpe={challenger_sharpe:.6g}")
    print(
        f"Champion: {champion_model.upper()} states={int(champion['states'])} "
        f"sharpe={float(champion['best_sharpe']):.6g}"
    )
    print(
        f"Challenger: {challenger_model.upper()} states={int(challenger['states'])} "
        f"sharpe={float(challenger['best_sharpe']):.6g}"
    )
    print(f"Wrote calibration report: {export_path.relative_to(ROOT)}")
    if args.no_promote:
        print("Live promotion skipped (--no-promote).")
    else:
        print(f"Promoted live model: {promote_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
