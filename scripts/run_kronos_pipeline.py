"""
run_kronos_pipeline.py — CLI for the Kronos foundation-model forecast layer.

Runs the Kronos autoregressive Transformer on historical OHLCV bars and
produces the standard RLM forecast-features CSV.  Optionally overlays an
HMM or Markov-switching regime model on top (mirroring the options available
in ``run_forecast_pipeline.py``).

Pre-requisites
--------------
::

    pip install -e ".[kronos]"
    # Requires torch>=2.0, einops, huggingface-hub, safetensors, tqdm.

Examples
--------
::

    # Basic: Kronos-small, 1-step ahead, 5 MC samples
    python scripts/run_kronos_pipeline.py --symbol SPY

    # Faster: stride=5, single sample, Kronos-mini (long context)
    python scripts/run_kronos_pipeline.py --symbol SPY \\
        --model NeoQuasar/Kronos-mini --max-context 512 \\
        --stride 5 --sample-count 1

    # With HMM overlay (6 states)
    python scripts/run_kronos_pipeline.py --symbol SPY --use-hmm --hmm-states 6

    # With Markov-switching overlay
    python scripts/run_kronos_pipeline.py --symbol SPY --use-markov --markov-states 3

    # Specific input/output paths
    python scripts/run_kronos_pipeline.py \\
        --bars data/raw/bars_QQQ.csv \\
        --out data/processed/kronos_forecast_QQQ.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.factors.pipeline import FactorPipeline

from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_forecast_features_csv
from rlm.forecasting.engines import HybridKronosForecastPipeline
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.kronos_forecast import KronosConfig, KronosForecastPipeline
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Kronos foundation-model forecast pipeline. "
            "Reads bars CSV → runs Kronos → writes forecast-features CSV."
        )
    )
    # I/O
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol (default: %(default)s)")
    p.add_argument(
        "--bars",
        default=None,
        help="Path to input bars CSV. Defaults to data/raw/bars_{SYMBOL}.csv.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to data/processed/kronos_forecast_{SYMBOL}.csv.",
    )

    # Kronos model
    p.add_argument(
        "--model",
        default="NeoQuasar/Kronos-small",
        help="HuggingFace model ID or local path (default: %(default)s).",
    )
    p.add_argument(
        "--tokenizer",
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="HuggingFace tokeniser ID or local path (default: %(default)s).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda:0', 'cpu'. Auto-detected if omitted.",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=512,
        help="Maximum context window length for Kronos (default: %(default)s).",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=200,
        help="Historical bars fed as context per inference call (default: %(default)s).",
    )
    p.add_argument(
        "--pred-len",
        type=int,
        default=1,
        help="Number of bars to predict per inference call (default: %(default)s).",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help=(
            "Independent stochastic draws for uncertainty estimation. "
            "1 = use realised-vol interval (fast). >1 = empirical quantiles (accurate, slower). "
            "Default: %(default)s."
        ),
    )
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold.")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling threshold (0=disabled).")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Run Kronos inference every N bars; predictions are filled forward in between. "
            "stride=1 (default) is most accurate; stride=pred-len is fastest."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Show Kronos tqdm progress bar.")

    # Regime overlay
    p.add_argument("--use-hmm", action="store_true", help="Overlay HMM regime model.")
    p.add_argument("--hmm-states", type=int, default=6, help="Number of HMM states.")
    p.add_argument(
        "--use-markov", action="store_true", help="Overlay Markov-switching regime model."
    )
    p.add_argument("--markov-states", type=int, default=3, help="Number of Markov states.")

    # RLM distributional config
    p.add_argument("--move-window", type=int, default=100)
    p.add_argument("--vol-window", type=int, default=100)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    symbol: str = args.symbol.upper()
    bars_path = Path(args.bars) if args.bars else ROOT / rel_bars_csv(symbol)
    out_path = (
        Path(args.out)
        if args.out
        else ROOT
        / rel_forecast_features_csv(symbol).replace("forecast_features", "kronos_forecast")
    )

    # ------------------------------------------------------------------
    # Load and enrich bars
    # ------------------------------------------------------------------
    print(f"[kronos] Loading bars: {bars_path}")
    df_raw = pd.read_csv(bars_path)
    df_enriched = prepare_bars_for_factors(df_raw)
    df_factors = FactorPipeline().run(df_enriched)

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    kronos_cfg = KronosConfig(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
        max_context=args.max_context,
        lookback=args.lookback,
        pred_len=args.pred_len,
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stride=args.stride,
        verbose=args.verbose,
    )
    rlm_cfg = ForecastConfig()

    hmm_config: HMMConfig | None = None
    markov_config: MarkovSwitchingConfig | None = None
    if args.use_hmm and args.use_markov:
        sys.exit("--use-hmm and --use-markov are mutually exclusive.")
    if args.use_hmm:
        hmm_config = HMMConfig(n_components=args.hmm_states)
    if args.use_markov:
        markov_config = MarkovSwitchingConfig(k_regimes=args.markov_states)

    if hmm_config is not None or markov_config is not None:
        print(
            f"[kronos] Using HybridKronosForecastPipeline "
            f"({'HMM' if hmm_config else 'Markov'} overlay)"
        )
        pipeline: KronosForecastPipeline | HybridKronosForecastPipeline = (
            HybridKronosForecastPipeline(
                kronos_config=kronos_cfg,
                rlm_config=rlm_cfg,
                move_window=args.move_window,
                vol_window=args.vol_window,
                hmm_config=hmm_config,
                markov_config=markov_config,
            )
        )
    else:
        print("[kronos] Using KronosForecastPipeline (no regime overlay)")
        pipeline = KronosForecastPipeline(
            config=kronos_cfg,
            rlm_config=rlm_cfg,
            move_window=args.move_window,
            vol_window=args.vol_window,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print(f"[kronos] Running forecast on {len(df_factors)} bars …")
    df_out = pipeline.run(df_factors)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path)
    print(f"[kronos] Forecast written to: {out_path}")

    forecast_source = df_out.get("forecast_source", pd.Series(["unknown"])).iloc[-1]
    print(f"[kronos] forecast_source (last bar): {forecast_source}")

    if "forecast_return_median" in df_out.columns:
        tail = df_out[["close", "forecast_return_median", "forecast_uncertainty"]].tail(5)
        print("\n[kronos] Last 5 bars:\n", tail.to_string())


if __name__ == "__main__":
    main()
