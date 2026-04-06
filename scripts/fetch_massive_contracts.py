#!/usr/bin/env python3
"""
Paginate Massive ``GET /v3/reference/options/contracts`` and save Parquet under:

    data/options/{UNDERLYING}/contracts/

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
from rlm.datasets.massive_options_parquet import fetch_option_contracts_to_parquet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--underlying", required=True, help="e.g. SPY")
    p.add_argument("--expiration-date", default=None, help="YYYY-MM-DD filter")
    p.add_argument("--contract-type", default=None, choices=("call", "put"))
    p.add_argument("--strike-price", type=float, default=None)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    params: dict = {"limit": args.limit}
    if args.expiration_date:
        params["expiration_date"] = args.expiration_date
    if args.contract_type:
        params["contract_type"] = args.contract_type
    if args.strike_price is not None:
        params["strike_price"] = args.strike_price

    client = MassiveClient()
    try:
        out = fetch_option_contracts_to_parquet(
            client,
            args.underlying,
            out_path=Path(args.out) if args.out else None,
            repo_root=ROOT,
            **params,
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
