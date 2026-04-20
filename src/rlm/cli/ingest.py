"""``rlm ingest`` — fetch and normalize market data into the data lake."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from rlm.cli.common import add_backend_arg, add_data_root_arg, add_profile_args, normalize_symbol
from rlm.core.run_manifest import RunManifest, write_run_manifest
from rlm.core.services.ingestion_service import IngestionRequest, IngestionService
from rlm.utils.logging import get_run_logger
from rlm.utils.run_id import generate_run_id


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rlm ingest", description="Fetch and normalize market data into the Parquet/DuckDB data lake.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--source", choices=["ibkr", "yfinance", "massive"], default="yfinance")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--no-options", action="store_true")
    add_data_root_arg(p)
    add_backend_arg(p)
    add_profile_args(p)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sym = normalize_symbol(args.symbol)
    run_id = generate_run_id("ingest")
    log = get_run_logger(__name__, run_id=run_id, command="ingest", symbol=sym, backend=args.backend, profile=args.profile)

    req = IngestionRequest(symbol=sym, source=args.source, start=args.start, end=args.end, interval=args.interval, fetch_options=not args.no_options, data_root=args.data_root)
    result = IngestionService().run(req)
    print(f"Wrote {result.bars_count} bars → {result.bars_path}")
    if result.chain_path:
        print(f"Wrote {result.chain_count} chain rows → {result.chain_path}")

    manifest = RunManifest(
        run_id=run_id,
        command="ingest",
        symbol=sym,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        backend=args.backend,
        profile=args.profile,
        config_summary={"source": args.source, "interval": args.interval},
        input_paths={"source": args.source},
        output_paths={"bars_path": str(result.bars_path), "chain_path": str(result.chain_path) if result.chain_path else ""},
        metrics={"bars_count": result.bars_count, "chain_count": result.chain_count},
    )
    manifest_path = write_run_manifest(manifest, data_root=args.data_root)
    log.info("ingest complete", extra={"stage": "complete", "success": True})
    print(f"Run manifest: {manifest_path}")
