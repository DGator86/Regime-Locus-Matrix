"""``rlm ingest`` — fetch and normalize market data into the data lake."""

from __future__ import annotations

import argparse

from rlm.cli.common import add_data_root_arg, normalize_symbol
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
    p.add_argument("--backend", choices=["auto", "csv", "lake"], default="auto")
    p.add_argument("--profile", default=None, help="Runtime profile name")
    p.add_argument("--config", default=None, help="Path to config JSON")
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    add_data_root_arg(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)

    log.info("ingest start  symbol=%s source=%s interval=%s", sym, args.source, args.interval)

    req = IngestionRequest(
        symbol=sym,
        source=args.source,
        start=args.start,
        end=args.end,
        interval=args.interval,
        fetch_options=not args.no_options,
        data_root=args.data_root,
        backend=args.backend,
        profile=args.profile,
        config_path=args.config,
        write_manifest=args.write_manifest,
    )

    result = IngestionService().run(req)
    log.info("ingest done  symbol=%s bars=%d path=%s", sym, result.bars_count, result.bars_path)
    print(f"Wrote {result.bars_count} bars → {result.bars_path}")
    if result.chain_path:
        print(f"Wrote {result.chain_count} chain rows → {result.chain_path}")
    if result.manifest_path:
        print(f"Manifest: {result.manifest_path}")
