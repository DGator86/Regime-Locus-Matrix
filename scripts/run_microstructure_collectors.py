#!/usr/bin/env python3
"""
Run the microstructure data collectors concurrently during market hours.

Starts two asyncio tasks:
  1. UnderlyingCollector  — streams 5-second OHLCV bars for each symbol
  2. OptionsCollector     — snapshots the full Greek surface every --interval seconds

Both write date-partitioned Parquet files to the microstructure data lake.

Examples::

    python scripts/run_microstructure_collectors.py --symbol SPY
    python scripts/run_microstructure_collectors.py --symbol SPY,QQQ --interval 10 --max-dte 45
    python scripts/run_microstructure_collectors.py --symbol SPY --no-options  # bars only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "microstructure_collectors.log"),
    ],
)
logger = logging.getLogger(__name__)


def _parse_symbols(s: str) -> list[str]:
    return [x.strip().upper() for x in s.replace(";", ",").split(",") if x.strip()]


async def _run_underlying(symbol: str, data_root: str, bar_size: str) -> None:
    from rlm.microstructure.collectors.underlying import UnderlyingCollector
    collector = UnderlyingCollector(symbol, data_root=data_root, bar_size=bar_size)
    await collector.stream()


async def _run_options(
    symbol: str,
    data_root: str,
    interval: float,
    strike_band: float,
    max_dte: int,
    min_oi: int,
) -> None:
    from rlm.microstructure.collectors.options import OptionsCollector
    collector = OptionsCollector(
        symbol,
        data_root=data_root,
        strike_band_pct=strike_band,
        max_dte=max_dte,
        min_open_interest=min_oi,
    )
    await collector.stream(interval=interval)


async def main(args: argparse.Namespace) -> None:
    symbols = _parse_symbols(args.symbols)
    data_root = str(ROOT / args.data_root)

    # Create log dir
    (ROOT / "logs").mkdir(exist_ok=True)

    tasks: list[asyncio.Task] = []
    for sym in symbols:
        tasks.append(asyncio.create_task(
            _run_underlying(sym, data_root, args.bar_size),
            name=f"underlying_{sym}",
        ))
        if not args.no_options:
            tasks.append(asyncio.create_task(
                _run_options(
                    sym, data_root, args.interval,
                    args.strike_band, args.max_dte, args.min_oi,
                ),
                name=f"options_{sym}",
            ))

    logger.info(
        "Started %d collector task(s) for %s (options=%s)",
        len(tasks), symbols, not args.no_options,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    logger.info("Shutdown signal received — cancelling tasks…")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All collector tasks stopped.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols", default="SPY", help="Comma-separated tickers")
    p.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between option chain snapshots (default: 5)",
    )
    p.add_argument(
        "--bar-size", default="5 secs",
        help="IBKR bar size for underlying stream (default: '5 secs')",
    )
    p.add_argument(
        "--strike-band", type=float, default=0.15,
        help="ATM band for option chain (default: 0.15 = ±15%%)",
    )
    p.add_argument("--max-dte", type=int, default=90, help="Max DTE for option chain (default: 90)")
    p.add_argument("--min-oi", type=int, default=10, help="Min open interest filter (default: 10)")
    p.add_argument(
        "--no-options", action="store_true",
        help="Skip the option chain collector (underlying bars only)",
    )
    p.add_argument(
        "--data-root", default="data/microstructure",
        help="Root of microstructure lake (relative to repo root)",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
