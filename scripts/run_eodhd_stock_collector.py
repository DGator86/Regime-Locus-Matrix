#!/usr/bin/env python3
"""EODHD 1m stock collector — rolling 5s/symbol, 09:00–16:30 ET.

Writes ``data/stocks/{SYM}/1m/{sym}_full_1m.parquet``. Universe pipeline reads these
via ``RLM_STOCK_BARS_SOURCE=eodhd`` (default). Options remain on Massive.

Examples::

    python3 scripts/run_eodhd_stock_collector.py
    python3 scripts/run_eodhd_stock_collector.py --once --backfill
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from rlm.data.eodhd_collector import (  # noqa: E402
    default_poll_symbols,
    refresh_symbol_intraday,
    run_eodhd_collector_loop,
)
from rlm.data.eodhd_stocks import fetch_intraday_lookback_days, is_eodhd_poll_window_open, load_eodhd_api_key  # noqa: E402
from rlm.data.stock_bars_provider import merge_bars_into_lake  # noqa: E402

DATA_ROOT = Path(os.environ.get("RLM_ROOT", str(ROOT))).expanduser().resolve()


def _backfill_all(symbols: tuple[str, ...], *, days: int) -> int:
    ok = 0
    for sym in symbols:
        print(f"[eodhd] backfill {sym} ({days}d)...", flush=True)
        fresh = fetch_intraday_lookback_days(sym, lookback_days=days)
        if fresh.empty:
            print(f"[eodhd] WARN: no bars for {sym}", flush=True)
            continue
        merge_bars_into_lake(sym, fresh, root=DATA_ROOT)
        print(f"[eodhd] {sym}: {len(fresh)} bars merged", flush=True)
        ok += 1
    return ok


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated tickers (default: LIQUID_TEN + SPY/QQQ first)",
    )
    ap.add_argument("--once", action="store_true", help="Poll each symbol once and exit")
    ap.add_argument(
        "--backfill",
        action="store_true",
        help="Full lookback pull (default 30d) before poll loop or --once",
    )
    ap.add_argument("--backfill-days", type=int, default=30)
    args = ap.parse_args()

    if not load_eodhd_api_key():
        print("EODHD_API_KEY missing — set in .env (scripts/set_eodhd_api_key.py)", file=sys.stderr)
        return 1

    if args.symbols.strip():
        symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    else:
        symbols = default_poll_symbols()

    if args.backfill:
        _backfill_all(symbols, days=int(args.backfill_days))

    if args.once:
        if not is_eodhd_poll_window_open():
            print("[eodhd] outside poll window (09:00–16:30 ET); --once still runs backfill only")
        for sym in symbols:
            try:
                n = refresh_symbol_intraday(sym, root=DATA_ROOT)
                print(f"[eodhd] {sym}: lake rows={n}", flush=True)
            except Exception as exc:
                print(f"[eodhd] {sym}: ERROR {exc}", flush=True)
        return 0

    run_eodhd_collector_loop(root=DATA_ROOT, symbols=symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
