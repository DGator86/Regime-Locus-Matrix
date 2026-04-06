#!/usr/bin/env python3
"""
Fetch Massive option chain snapshot and append to data/raw/option_chain_{SYMBOL}.csv.

Run daily (or after the close) with the same --as-of date to build real option history for
walk-forward / backtests. Use --replace-same-day to overwrite an earlier pull on that calendar date.

Requires MASSIVE_API_KEY in the environment (.env supported via python-dotenv if installed).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rlm.data.massive import MassiveClient
from rlm.data.massive_option_chain import massive_option_chain_from_client
from rlm.datasets.paths import DEFAULT_SYMBOL, rel_option_chain_csv
from rlm.datasets.option_history import (
    merge_option_chain_history,
    read_option_chain_csv,
    write_option_chain_csv,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Underlying symbol (default {DEFAULT_SYMBOL})",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default data/raw/option_chain_{SYMBOL}.csv)",
    )
    p.add_argument(
        "--as-of",
        default=None,
        help="Snapshot timestamp as YYYY-MM-DD (default: today UTC date at 00:00)",
    )
    p.add_argument(
        "--replace-same-day",
        action="store_true",
        help="Remove existing rows with the same calendar date as --as-of before appending",
    )
    args = p.parse_args()

    if not os.environ.get("MASSIVE_API_KEY"):
        print("MASSIVE_API_KEY is not set.", file=sys.stderr)
        return 1

    sym = args.symbol.upper().strip()
    out_path = Path(args.out) if args.out else ROOT / rel_option_chain_csv(sym)

    if args.as_of:
        as_of = pd.Timestamp(args.as_of).normalize()
    else:
        as_of = pd.Timestamp.utcnow().normalize()

    client = MassiveClient()
    chain = massive_option_chain_from_client(
        client, sym, timestamp=as_of, limit=250
    )

    if chain.empty:
        print("No option chain rows returned from Massive.", file=sys.stderr)
        return 1

    existing = read_option_chain_csv(out_path)
    replace_dt = as_of if args.replace_same_day else None
    merged = merge_option_chain_history(
        existing if not existing.empty else None,
        chain,
        replace_calendar_date=replace_dt,
    )
    write_option_chain_csv(merged, out_path)

    print(
        f"Wrote {len(merged)} rows to {out_path} "
        f"(+{len(chain)} from snapshot as_of={as_of.date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
