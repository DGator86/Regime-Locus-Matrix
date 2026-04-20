"""``rlm ingest`` — fetch and normalize market data into the data lake."""

from __future__ import annotations

import argparse

from rlm.core.services.ingestion_service import IngestionRequest, IngestionService
from rlm.utils.logging import get_logger

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm ingest",
        description="Fetch and normalize market data into the Parquet/DuckDB data lake.",
    )
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. SPY)")
    p.add_argument(
        "--source",
        choices=["ibkr", "yfinance", "massive"],
        default="yfinance",
        help="Data provider (default: yfinance)",
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    p.add_argument("--no-options", action="store_true", help="Skip option chain fetch")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = args.symbol.upper().strip()

    log.info("Starting ingestion: symbol=%s source=%s interval=%s", sym, args.source, args.interval)

    req = IngestionRequest(
        symbol=sym,
        source=args.source,
        start=args.start,
        end=args.end,
        interval=args.interval,
        fetch_options=not args.no_options,
    )

    result = IngestionService().run(req)
    log.info("Ingestion complete: %d bars written to %s", result.bars_count, result.bars_path)
    if result.chain_path:
        log.info("Option chain: %d rows written to %s", result.chain_count, result.chain_path)
