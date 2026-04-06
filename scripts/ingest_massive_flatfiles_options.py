#!/usr/bin/env python3
"""
Bulk-ingest **Massive options Flat Files** (S3-compatible gzip CSV) into the data lake.

Writes one Parquet per calendar day under:

    data/options/{UNDERLYING}/flatfiles/{trades|quotes|day_aggs|minute_aggs}/{date}.parquet

Requires S3 flat-file credentials (not REST-only ``MASSIVE_API_KEY``). Install:

    pip install -e ".[flatfiles]"

Prefer ``--underlying SPY`` (filters to tickers starting with ``O:SPY``). Full unfiltered
daily files are enormous; use ``--allow-full-file`` only if you intend to read the whole gzip.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _normalize_dataset_arg(raw: str) -> str:
    x = raw.strip().lower().replace("-", "_")
    aliases = {
        "trades": "trades",
        "quotes": "quotes",
        "day_aggs": "day_aggs",
        "dayaggs": "day_aggs",
        "minute_aggs": "minute_aggs",
        "minuteaggs": "minute_aggs",
    }
    if x not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unknown dataset {raw!r}; use trades, quotes, day-aggs, minute-aggs"
        )
    return aliases[x]


def _underlying_for_lake_path(args: argparse.Namespace, prefixes: list[str]) -> str:
    if args.allow_full_file and not prefixes:
        return "BULK"
    if args.underlying:
        return str(args.underlying).strip().upper()
    if len(prefixes) == 1:
        m = re.match(r"^O:([A-Za-z]+)", str(prefixes[0]).strip().upper())
        if m:
            return m.group(1).upper()
    raise ValueError(
        "Pass --underlying SPY (or a single --ticker-prefix like O:SPY) so output paths "
        "use data/options/{UNDERLYING}/flatfiles/..."
    )


def main() -> int:
    from rlm.data.massive_flatfiles import (
        FlatfileDataset,
        download_s3_object,
        load_massive_flatfiles_config,
        options_flatfile_object_key,
    )
    from rlm.datasets.data_lake import options_flatfile_daily_parquet
    from rlm.datasets.flatfiles_ingest import gzip_csv_to_filtered_parquet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=_normalize_dataset_arg,
        required=True,
        help="trades | quotes | day-aggs | minute-aggs",
    )
    p.add_argument("--from-date", type=_parse_date, required=True)
    p.add_argument("--to-date", type=_parse_date, required=True)
    p.add_argument(
        "--underlying",
        default=None,
        help="e.g. SPY — filters rows where ticker starts with O:SPY (recommended)",
    )
    p.add_argument(
        "--ticker-prefix",
        action="append",
        default=[],
        help="Repeatable; e.g. O:SPY. Overrides implicit prefix from --underlying if set.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print S3 keys and output paths only",
    )
    p.add_argument(
        "--staging-dir",
        default=None,
        help="Download .csv.gz here and keep after ingest (default: temp file, deleted)",
    )
    p.add_argument("--chunksize", type=int, default=400_000)
    p.add_argument(
        "--allow-full-file",
        action="store_true",
        help="Do not require ticker prefix filters (downloads full daily file → huge Parquet)",
    )
    args = p.parse_args()

    if args.from_date > args.to_date:
        print("--from-date must be <= --to-date", file=sys.stderr)
        return 2

    prefixes: list[str] = [x for x in args.ticker_prefix if str(x).strip()]
    if args.underlying and not prefixes:
        u = str(args.underlying).strip().upper()
        prefixes = [f"O:{u}"]

    if not prefixes and not args.allow_full_file:
        print(
            "Refusing unfiltered ingest: pass --underlying SPY and/or --ticker-prefix O:SPY "
            "or use --allow-full-file (very large).",
            file=sys.stderr,
        )
        return 2

    if not prefixes and args.allow_full_file:
        print(
            "WARNING: ingesting full daily flat file with no ticker filter — expect huge "
            "memory/disk use.",
            file=sys.stderr,
        )

    ds: FlatfileDataset = args.dataset  # type: ignore[assignment]

    try:
        cfg = load_massive_flatfiles_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    try:
        underlying_for_path = _underlying_for_lake_path(args, prefixes)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    staging = Path(args.staging_dir).resolve() if args.staging_dir else None
    if staging:
        staging.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        from botocore.exceptions import ClientError

    for d in _daterange(args.from_date, args.to_date):
        key = options_flatfile_object_key(ds, d, key_style=cfg.key_style)
        out = options_flatfile_daily_parquet(underlying_for_path, ds, d, root=ROOT)
        if args.dry_run:
            print(f"s3://{cfg.bucket}/{key} -> {out}")
            continue

        if staging:
            gz_path = staging / f"{ds}_{d.isoformat()}.csv.gz"
            gz_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False)
            gz_path = Path(tmp.name)
            tmp.close()

        try:
            try:
                download_s3_object(cfg.bucket, key, str(gz_path), cfg=cfg)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey", "NotFound"):
                    print(f"Missing: s3://{cfg.bucket}/{key}", file=sys.stderr)
                else:
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
                try:
                    gz_path.unlink(missing_ok=True)
                except OSError:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
