from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.features.factors.pipeline import FactorPipeline

from rlm.data.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_bars_csv, rel_option_chain_csv
from rlm.forecasting.distribution import estimate_distribution
from rlm.forecasting.probabilistic import (
    DEFAULT_PROB_FEATURE_COLUMNS,
    build_probabilistic_feature_frame,
)
from rlm.types.forecast import ForecastConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a quantile-based probabilistic forecast model.")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--bars", default=None, help="Bars CSV relative to repo root.")
    p.add_argument("--chain", default=None, help="Option chain CSV relative to repo root.")
    p.add_argument(
        "--out",
        default="models/probabilistic_forecast.json",
        help="Model artifact path relative to repo root.",
    )
    p.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantiles to fit (must include 0.5).",
    )
    p.add_argument(
        "--feature-cols",
        default="",
        help="Optional comma-separated feature override. Default uses built-in probabilistic features.",
    )
    p.add_argument("--no-vix", action="store_true", help="Skip yfinance VIX/VVIX enrichment.")
    return p.parse_args()


def _parse_quantiles(raw: str) -> list[float]:
    quantiles = sorted(float(x.strip()) for x in raw.split(",") if x.strip())
    if 0.5 not in quantiles:
        raise SystemExit("--quantiles must include 0.5 for the median forecast.")
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise SystemExit("All quantiles must be strictly between 0 and 1.")
    return quantiles


def main() -> None:
    args = parse_args()
    sym = str(args.symbol).upper().strip()
    bars_path = ROOT / (args.bars or rel_bars_csv(sym))
    chain_path = ROOT / (args.chain or rel_option_chain_csv(sym))
    out_path = ROOT / args.out

    if not bars_path.is_file():
        raise SystemExit(f"Bars file not found: {bars_path}")

    bars = pd.read_csv(bars_path, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    chain = pd.read_csv(chain_path, parse_dates=["timestamp", "expiry"]) if chain_path.is_file() else None

    enriched = prepare_bars_for_factors(bars, chain, underlying=sym, attach_vix=not args.no_vix)
    factors = FactorPipeline().run(enriched)
    base = estimate_distribution(factors, config=ForecastConfig())

    if args.feature_cols.strip():
        feature_cols = tuple(c.strip() for c in args.feature_cols.split(",") if c.strip())
    else:
        feature_cols = tuple(c for c in DEFAULT_PROB_FEATURE_COLUMNS if c in base.columns)

    feature_frame = build_probabilistic_feature_frame(base, feature_columns=feature_cols)
    target = base["close"].pct_change().shift(-1).rename("forward_return")

    train = feature_frame.copy()
    train["forward_return"] = target
    train = train.dropna(subset=["forward_return"])
    if train.empty:
        raise SystemExit("No rows available after target alignment.")

    x = sm.add_constant(train.loc[:, list(feature_cols)], has_constant="add")
    y = train["forward_return"].astype(float)

    quantiles = _parse_quantiles(args.quantiles)
    intercepts: list[float] = []
    coefficients: list[list[float]] = []

    for q in quantiles:
        result = sm.QuantReg(y, x).fit(q=q, max_iter=5_000)
        intercepts.append(float(result.params["const"]))
        coefficients.append([float(result.params[col]) for col in feature_cols])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "quantiles": quantiles,
        "feature_columns": list(feature_cols),
        "intercepts": intercepts,
        "coefficients": coefficients,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
