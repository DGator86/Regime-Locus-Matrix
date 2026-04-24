#!/usr/bin/env python3
"""
Run an intraday (5-minute) backtest over a recent window (default: last 3 calendar months).

**Demo (no IBKR):** synthetic 5m bars + synthetic option chain (same engine as daily ``run_backtest.py``).

**Live data:** ``--fetch-ibkr`` pulls 5m RTH bars from TWS/Gateway (chunked). Option chain is still
synthetic-from-bars unless you supply ``--chain`` pointing at a CSV aligned to bar timestamps.

Forecast rolling windows default to **390 bars** (~5 RTH sessions of 5m bars), not 100 days.

Example:

    python scripts/run_backtest_5m.py --demo --months 3 --symbol SPY --no-vix
    python scripts/run_backtest_5m.py --fetch-ibkr --months 3 --symbol SPY --no-vix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.backtest.engine import BacktestEngine
from rlm.datasets.backtest_data import (
    fetch_ibkr_5m_bars_range,
    synthetic_5m_bars_range,
    synthetic_option_chain_intraday_from_bars,
)
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.datasets.paths import DEFAULT_SYMBOL
from rlm.features.factors.pipeline import FactorPipeline
from rlm.forecasting.hmm import HMMConfig
from rlm.forecasting.markov_switching import MarkovSwitchingConfig
from rlm.forecasting.engines import ForecastPipeline, HybridForecastPipeline, HybridMarkovForecastPipeline
from rlm.roee.pipeline import ROEEConfig
from rlm.types.forecast import ForecastConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--months", type=int, default=3, help="Calendar months of history ending today")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic 5m bars (no TWS). Mutually exclusive with --fetch-ibkr / --bars-in.",
    )
    p.add_argument(
        "--fetch-ibkr",
        action="store_true",
        help="Download 5m bars from IBKR (TWS/Gateway must be running).",
    )
    p.add_argument(
        "--bars-in",
        default="",
        help="CSV with timestamp + OHLCV (+ vwap). Skips --demo and --fetch-ibkr.",
    )
    p.add_argument(
        "--chain",
        default="",
        help="Option chain CSV (must have timestamp matching each bar). Default: synthetic from bars.",
    )
    p.add_argument("--use-hmm", action="store_true")
    p.add_argument("--hmm-states", type=int, default=6)
    p.add_argument("--use-markov", action="store_true", help="Use Markov-switching regime model.")
    p.add_argument("--no-vix", action="store_true", help="Skip yfinance VIX/VVIX during enrichment")
    p.add_argument(
        "--move-window",
        type=int,
        default=390,
        help="Forecast baseline rolling length in **bars** (default 390 ~ 5 RTH days at 5m).",
    )
    p.add_argument(
        "--out-tag",
        default="5m",
        help="Output filenames: backtest_equity_{SYM}_{tag}.csv",
    )
    return p.parse_args()


def _load_chain(path: str, bars: pd.DataFrame, sym: str) -> pd.DataFrame:
    if path:
        p = ROOT / path
        if not p.is_file():
            raise FileNotFoundError(f"Chain file not found: {p}")
        return pd.read_csv(p, parse_dates=["timestamp", "expiry"])
    # One greek grid per session day, stamped onto each 5m bar (fast vs per-bar BS).
    return synthetic_option_chain_intraday_from_bars(bars, underlying=sym)


def main() -> int:
    args = _parse_args()
    sym = str(args.symbol).upper().strip()

    if args.demo + args.fetch_ibkr + bool(args.bars_in) != 1:
        print("Choose exactly one of: --demo, --fetch-ibkr, or --bars-in PATH", file=sys.stderr)
        return 2

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=int(args.months))

    if args.demo:
        bars = synthetic_5m_bars_range(start, end)
    elif args.fetch_ibkr:
        print(f"IBKR 5m {sym} {start.date()} .. {end.date()} …")
        bars = fetch_ibkr_5m_bars_range(sym, start=start, end=end)
    else:
        p = Path(args.bars_in)
        if not p.is_file():
            print(f"--bars-in not found: {p}", file=sys.stderr)
            return 2
        bars = pd.read_csv(p, parse_dates=["timestamp"])
        bars = bars.sort_values("timestamp").set_index("timestamp")

    if bars.empty:
        print("No bars in range; check dates or IBKR session.", file=sys.stderr)
        return 1

    chain = _load_chain(args.chain, bars, sym)

    mw = max(50, int(args.move_window))
    bars = prepare_bars_for_factors(bars, chain, underlying=sym, attach_vix=not args.no_vix)

    features = FactorPipeline().run(bars)
    fc = ForecastConfig(
        drift_gamma_alpha=0.65,
        sigma_floor=1e-4,
        direction_neutral_threshold=0.3,
    )
    if args.use_hmm:
        features = HybridForecastPipeline(
            config=fc,
            move_window=mw,
            vol_window=mw,
            hmm_config=HMMConfig(n_states=args.hmm_states),
        ).run(features)
    elif args.use_markov:
        features = HybridMarkovForecastPipeline(
            config=fc,
            move_window=mw,
            vol_window=mw,
            markov_config=MarkovSwitchingConfig(),
        ).run(features)
    else:
        features = ForecastPipeline(config=fc, move_window=mw, vol_window=mw).run(features)

    engine = BacktestEngine(
        initial_capital=100_000.0,
        contract_multiplier=100,
        strike_increment=5.0,
        underlying_symbol=sym,
        quantity_per_trade=1,
        roee_config=ROEEConfig() if (args.use_hmm or args.use_markov) else None,
    )

    equity_frame, trades_frame, summary = engine.run(features, chain)

    print("5m backtest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.out_tag).strip() or "5m"
    equity_path = out_dir / f"backtest_equity_{sym}_{tag}.csv"
    trades_path = out_dir / f"backtest_trades_{sym}_{tag}.csv"
    equity_frame.to_csv(equity_path)
    trades_frame.to_csv(trades_path, index=False)
    print(f"Wrote {equity_path.relative_to(ROOT)} and {trades_path.relative_to(ROOT)}.")
    print(f"Bars used: {len(bars)} (5m RTH-style).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
