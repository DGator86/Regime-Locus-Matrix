#!/usr/bin/env python3
"""
Massive option trades (paginated) → Parquet under:

    data/options/{UNDERLYING}/trades/

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
from rlm.datasets.massive_options_parquet import fetch_option_trades_to_parquet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--option-ticker", required=True)
    p.add_argument("--underlying-path", required=True, help="e.g. SPY")
    p.add_argument("--timestamp-gte", required=True, help="ISO8601 e.g. 2026-03-20T13:30:00Z")
    p.add_argument("--timestamp-lt", required=True)
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    client = MassiveClient()
    try:
        out = fetch_option_trades_to_parquet(
            client,
            args.option_ticker,
            underlying_for_path=args.underlying_path,
            ts_gte=args.timestamp_gte,
            ts_lt=args.timestamp_lt,
            out_path=Path(args.out) if args.out else None,
            repo_root=ROOT,
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
