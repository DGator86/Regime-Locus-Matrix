"""``rlm ingest`` — fetch and normalize market data into the data lake."""

from __future__ import annotations

import argparse

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args, normalize_symbol
from rlm.core.services.ingestion_service import IngestionRequest, IngestionService


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rlm ingest", description="Fetch and normalize market data into local storage."
    )
    p.add_argument("--symbol", required=True, help="Ticker symbol (e.g. SPY)")
    p.add_argument(
        "--source", choices=["ibkr", "yfinance"], default="yfinance", help="Data provider"
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    p.add_argument("--no-options", action="store_true", help="Skip option chain fetch")
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    req = IngestionRequest(
        symbol=normalize_symbol(args.symbol),
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
    print(f"Wrote {result.bars_count} bars -> {result.bars_path}")
    if result.chain_path:
        print(f"Wrote {result.chain_count} chain rows -> {result.chain_path}")
    if result.manifest_path:
        print(f"Manifest: {result.manifest_path}")
