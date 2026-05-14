from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.features.factors.pipeline import FactorPipeline

from rlm.data.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_features_csv, rel_option_chain_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build factor features from daily bars (default: data/raw/bars_{SYMBOL}.csv)."
    )
    p.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Ticker for default paths (default {DEFAULT_SYMBOL})",
    )
    p.add_argument(
        "--bars",
        default=None,
        help="Bars CSV relative to repo root (default: data/raw/bars_{SYMBOL}.csv)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV relative to repo root (default: data/processed/features_{SYMBOL}.csv)",
    )
    p.add_argument(
        "--chain",
        default=None,
        help="Option chain CSV for dealer/liquidity enrichment (default: data/raw/option_chain_{SYMBOL}.csv if present)",
    )
    p.add_argument(
        "--no-vix",
        action="store_true",
        help="Skip yfinance ^VIX/^VVIX attachment (offline runs).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sym = str(args.symbol).upper().strip()
    bars_rel = args.bars or rel_bars_csv(sym)
    out_rel = args.out or rel_features_csv(sym)
    bars_path = ROOT / bars_rel
    if not bars_path.is_file():
        raise SystemExit(
            f"Bars file not found: {bars_path}\n"
            "Fetch with: python scripts/build_rolling_backtest_dataset.py --fetch-ibkr --symbol "
            f"{sym} --start YYYY-MM-DD\n"
            "Or demo: python scripts/build_rolling_backtest_dataset.py --demo"
        )

    df = pd.read_csv(bars_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    chain_rel = args.chain or rel_option_chain_csv(sym)
    chain_path = ROOT / chain_rel
    opch: pd.DataFrame | None = None
    if chain_path.is_file():
        opch = pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"])

    df = prepare_bars_for_factors(df, opch, underlying=sym, attach_vix=not args.no_vix)

    pipeline = FactorPipeline()
    features = pipeline.run(df)

    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path)
    print(features.tail(5)[["S_D", "S_V", "S_L", "S_G"]])
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
