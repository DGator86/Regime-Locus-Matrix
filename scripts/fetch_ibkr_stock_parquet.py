#!/usr/bin/env python3
"""
Pull underlying stock bars from IBKR (ibapi) and write Parquet under:

    data/stocks/{SYMBOL}/1d/  and/or  data/stocks/{SYMBOL}/1m/

Requires TWS/Gateway and: pip install 'regime-locus-matrix[ibkr]' pyarrow
(.env: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)

For an ib_insync-based variant, see scripts/examples/ib_insync_fetch_stock_example.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.ingestion.writers import write_ibkr_stock_parquet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbol", help="US stock symbol, e.g. SPY")
    p.add_argument("--duration", default="2 Y", help='IB duration string, e.g. "2 Y", "10 D"')
    p.add_argument("--bar-size", default="1 day", help='e.g. "1 day", "1 min"')
    p.add_argument(
        "--duration-slug",
        default=None,
        help="Filename token (default: derived from --duration)",
    )
    p.add_argument(
        "--interval",
        choices=("1d", "1m"),
        default="1d",
        help="Folder layout under data/stocks/{SYM}/",
    )
    p.add_argument("--out", default=None, help="Override output Parquet path")
    p.add_argument(
        "--end-datetime", default="", dest="end_datetime", help="IB end anchor; empty = now"
    )
    args = p.parse_args()

    slug = args.duration_slug
    if not slug:
        slug = args.duration.replace(" ", "").lower()

    try:
        path = write_ibkr_stock_parquet(
            args.symbol,
            duration=args.duration,
            bar_size=args.bar_size,
            duration_slug=slug,
            out_path=Path(args.out) if args.out else None,
            interval_dir="1m" if args.interval == "1m" else "1d",
            repo_root=ROOT,
            end_datetime=args.end_datetime,
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        return 2
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
