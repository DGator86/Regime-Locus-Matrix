#!/usr/bin/env python3
"""
Massive option aggregates (daily or minute) → Parquet under:

    data/options/{UNDERLYING}/bars_1d/  or  bars_1m/

Ticker format: ``O:SPY260619C00650000`` (OCC Massive options ticker).

Requires MASSIVE_API_KEY and: pip install pyarrow
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

from rlm.data.massive import MassiveClient
from rlm.datasets.massive_options_parquet import fetch_option_aggs_to_parquet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--option-ticker", required=True, help="e.g. O:SPY260619C00650000")
    p.add_argument("--underlying-path", required=True, help="Folder name under data/options/, e.g. SPY")
    p.add_argument("--multiplier", type=int, default=1)
    p.add_argument("--timespan", choices=("minute", "day", "hour"), default="day")
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--adjusted", default="true")
    p.add_argument("--sort", default="asc")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    client = MassiveClient()
    try:
        out = fetch_option_aggs_to_parquet(
            client,
            args.option_ticker,
            multiplier=args.multiplier,
            timespan=args.timespan,
            from_date=args.from_date,
            to_date=args.to_date,
            underlying_for_path=args.underlying_path,
            out_path=Path(args.out) if args.out else None,
            repo_root=ROOT,
            adjusted=args.adjusted,
            sort=args.sort,
            limit=args.limit,
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        return 2
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
