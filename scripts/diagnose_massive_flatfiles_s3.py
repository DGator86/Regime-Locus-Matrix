#!/usr/bin/env python3
"""
Check Massive flat-file S3 credentials: ListObjects vs GetObject.

If **list** works but **get** returns 403 for the same key, the client signing/URL shape is
usually fine; Massive must grant **s3:GetObject** (or equivalent) on your flat-file access key.
Contact Massive support with this script's output.

Usage (from repo root, with .env):

    python scripts/diagnose_massive_flatfiles_s3.py
    python scripts/diagnose_massive_flatfiles_s3.py --key us_stocks_sip/day_aggs_v1/2020/01/2020-01-02.csv.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    from botocore.exceptions import ClientError

    from rlm.data.massive_flatfiles import boto3_s3_client, load_massive_flatfiles_config

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--key",
        default=None,
        help="Full object key to fetch (default: first key under us_options_opra/trades_v1/)",
    )
    args = p.parse_args()

    try:
        cfg = load_massive_flatfiles_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    client = boto3_s3_client(cfg)
    prefix = "us_options_opra/trades_v1/2014/06/"
    try:
        lst = client.list_objects_v2(Bucket=cfg.bucket, Prefix=prefix, MaxKeys=3)
        n = lst.get("KeyCount", 0)
        print(
            f"list_objects_v2 bucket={cfg.bucket!r} prefix={prefix!r} -> KeyCount={n}", flush=True
        )
        for o in lst.get("Contents", []) or []:
            print(" ", o["Key"], flush=True)
    except ClientError as e:
        print("list_objects_v2 FAILED:", e.response.get("Error", {}), file=sys.stderr)
        return 1

    key = args.key
    if not key and lst.get("Contents"):
        key = lst["Contents"][0]["Key"]
    if not key:
        print("No key to test GetObject; pass --key", file=sys.stderr)
        return 1

    print(f"get_object key={key!r} ...", flush=True)
    try:
        r = client.get_object(Bucket=cfg.bucket, Key=key)
        blob = r["Body"].read(256)
        print(
            f"get_object OK, first {len(blob)} bytes read, ContentLength={r.get('ContentLength')}"
        )
    except ClientError as e:
        err = e.response.get("Error", {})
        print("get_object FAILED:", err, file=sys.stderr)
        print(
            "\nIf ListObjects succeeded but GetObject is 403, ask Massive to enable object read "
            "for your flat-file S3 credentials (same symptom as Polygon flat files + some SDKs).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
