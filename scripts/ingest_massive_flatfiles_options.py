#!/usr/bin/env python3
"""Bulk-ingest Massive options Flat Files (S3 gzip CSV) into the local data lake."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rlm.data.lake import options_flatfile_daily_parquet
from rlm.data.massive_flatfiles import (
    FlatfileDataset,
    download_s3_object,
    load_massive_flatfiles_config,
    options_flatfile_object_key,
)
from rlm.datasets.flatfiles_ingest import gzip_csv_to_filtered_parquet
from rlm.ingestion.adapters.flatfiles import (
    date_range,
    normalize_dataset_arg,
    parse_date,
    underlying_for_lake_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=normalize_dataset_arg, required=True)
    p.add_argument("--from-date", type=parse_date, required=True)
    p.add_argument("--to-date", type=parse_date, required=True)
    p.add_argument("--underlying", default=None)
    p.add_argument("--ticker-prefix", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--chunksize", type=int, default=400_000)
    p.add_argument("--allow-full-file", action="store_true")
    args = p.parse_args()

    if args.from_date > args.to_date:
        print("--from-date must be <= --to-date", file=sys.stderr)
        return 2

    prefixes = [x for x in args.ticker_prefix if str(x).strip()]
    if args.underlying and not prefixes:
        prefixes = [f"O:{str(args.underlying).strip().upper()}"]

    if not prefixes and not args.allow_full_file:
        print("Refusing unfiltered ingest unless --allow-full-file is set.", file=sys.stderr)
        return 2

    ds: FlatfileDataset = args.dataset  # type: ignore[assignment]

    try:
        cfg = load_massive_flatfiles_config()
        underlying = underlying_for_lake_path(
            underlying=args.underlying,
            allow_full_file=args.allow_full_file,
            prefixes=prefixes,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    staging = Path(args.staging_dir).resolve() if args.staging_dir else None
    if staging:
        staging.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        from botocore.exceptions import ClientError

    for d in date_range(args.from_date, args.to_date):
        key = options_flatfile_object_key(ds, d, key_style=cfg.key_style)
        out = options_flatfile_daily_parquet(underlying, ds, d, root=ROOT)
        if args.dry_run:
            print(f"s3://{cfg.bucket}/{key} -> {out}")
            continue

        gz_path = (
            (staging / f"{ds}_{d.isoformat()}.csv.gz")
            if staging
            else Path(tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False).name)
        )
        try:
            try:
                download_s3_object(cfg.bucket, key, str(gz_path), cfg=cfg)
            except ClientError as e:
                print(f"S3 error for {key}: {e}", file=sys.stderr)
                continue

            n = gzip_csv_to_filtered_parquet(
                gz_path,
                out,
                ticker_prefixes=prefixes if prefixes else None,
                chunksize=args.chunksize,
            )
            print(f"{d.isoformat()} {ds} rows={n} -> {out}")
        finally:
            if not staging:
                gz_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
