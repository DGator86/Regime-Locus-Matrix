#!/usr/bin/env python3
"""
**Full-universe sweep** while the market is open: pull fresh **daily** history from IBKR per
symbol, run **factors → state matrix → forecast bands → ROEE** on the latest bar, and print a table.

For **Massive option chains + matched legs + entry/stop/trail plan + IBKR combo JSON**, use
``scripts/run_universe_options_pipeline.py`` instead.

Uses :data:`rlm.data.liquidity_universe.LIQUID_UNIVERSE` by default (Magnificent 7 + SPY + QQQ).

Examples::

    python scripts/analyze_universe_live.py
    python scripts/analyze_universe_live.py --symbols AAPL,MSFT,SPY --duration \"120 D\" --ibkr-delay 0.4
    python scripts/analyze_universe_live.py --no-vix --out data/processed/universe_scan.csv

Requires TWS / IB Gateway (``pip install -e '.[ibkr]'``) and ``IBKR_*`` in ``.env``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.ibkr_stocks import fetch_historical_stock_bars
from rlm.data.liquidity_universe import LIQUID_UNIVERSE
from rlm.datasets.bars_enrichment import prepare_bars_for_factors
from rlm.factors.pipeline import FactorPipeline
from rlm.forecasting.pipeline import ForecastPipeline
from rlm.roee.decision import select_trade_for_row
from rlm.scoring.state_matrix import classify_state_matrix


def _parse_symbols(s: str) -> list[str]:
    parts = [x.strip().upper() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _one_symbol(
    sym: str,
    *,
    duration: str,
    move_window: int,
    vol_window: int,
    strike_increment: float,
    attach_vix: bool,
) -> dict[str, object]:
    bars = fetch_historical_stock_bars(
        sym,
        duration=duration,
        bar_size="1 day",
        timeout_sec=120.0,
    )
    if bars.empty:
        return {
            "symbol": sym,
            "close": None,
            "sigma": None,
            "regime_key": "",
            "S_D": None,
            "S_V": None,
            "S_L": None,
            "S_G": None,
            "action": "skip",
            "strategy": "",
            "size_fraction": None,
            "rationale": "",
            "error": "no bars",
        }

    df = bars.sort_values("timestamp").set_index("timestamp")
    df = prepare_bars_for_factors(df, option_chain=None, underlying=sym, attach_vix=attach_vix)

    feats = FactorPipeline().run(df)
    feats = classify_state_matrix(feats)

    forecast = ForecastPipeline(move_window=move_window, vol_window=vol_window)
    out = forecast.run(feats)
    out = out.copy()
    out["has_major_event"] = False

    last = out.iloc[-1]
    d = select_trade_for_row(last, strike_increment=strike_increment)

    return {
        "symbol": sym,
        "close": float(last["close"]),
        "sigma": float(last["sigma"]),
        "regime_key": str(last.get("regime_key", "")),
        "S_D": float(last["S_D"]) if pd.notna(last["S_D"]) else None,
        "S_V": float(last["S_V"]) if pd.notna(last["S_V"]) else None,
        "S_L": float(last["S_L"]) if pd.notna(last["S_L"]) else None,
        "S_G": float(last["S_G"]) if pd.notna(last["S_G"]) else None,
        "action": d.action,
        "strategy": d.strategy_name or "",
        "size_fraction": d.size_fraction,
        "rationale": (d.rationale or "")[:80],
        "error": "",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--symbols",
        default=",".join(LIQUID_UNIVERSE),
        help="Comma-separated tickers (default: LIQUID_UNIVERSE)",
    )
    p.add_argument("--duration", default="180 D", help="IBKR historical duration string")
    p.add_argument("--move-window", type=int, default=100, help="Distribution move baseline window")
    p.add_argument("--vol-window", type=int, default=100, help="Vol baseline window")
    p.add_argument("--strike-increment", type=float, default=1.0, help="ROEE strike grid (use 0.5 for sub-$200 names)")
    p.add_argument("--ibkr-delay", type=float, default=0.35, help="Seconds between IBKR requests (pacing)")
    p.add_argument("--no-vix", action="store_true", help="Skip ^VIX/^VVIX (faster, less macro context)")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV path under repo root")
    args = p.parse_args()

    syms = _parse_symbols(args.symbols)
    rows: list[dict[str, object]] = []

    for i, sym in enumerate(syms):
        if i:
            time.sleep(max(0.0, args.ibkr_delay))
        try:
            row = _one_symbol(
                sym,
                duration=args.duration,
                move_window=args.move_window,
                vol_window=args.vol_window,
                strike_increment=args.strike_increment,
                attach_vix=not args.no_vix,
            )
        except Exception as e:
            row = {"symbol": sym, "error": str(e)[:200]}
        rows.append(row)
        err = row.get("error", "")
        act = row.get("action", "")
        print(f"{sym}: {'ERR ' + str(err) if err else act} {row.get('strategy', '')}")

    table = pd.DataFrame(rows)
    cols = [
        "symbol",
        "close",
        "action",
        "strategy",
        "size_fraction",
        "regime_key",
        "S_D",
        "S_V",
        "error",
    ]
    cols = [c for c in cols if c in table.columns]
    print("\n" + table[cols].to_string(index=False))

    if args.out:
        out_path = ROOT / args.out if not args.out.is_absolute() else args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
