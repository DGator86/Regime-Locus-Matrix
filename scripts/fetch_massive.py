"""CLI smoke test for Massive API (requires MASSIVE_API_KEY)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rlm.data.massive import MassiveClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch JSON from Massive REST API")
    parser.add_argument(
        "ticker", nargs="?", default="SPY", help="Underlying for options snapshot / stock aggs"
    )
    parser.add_argument(
        "--endpoint",
        choices=(
            "option-snapshot",
            "stock-aggs",
            "stocks-trades",
            "stocks-quotes",
            "raw",
        ),
        default="option-snapshot",
    )
    parser.add_argument(
        "--path",
        default="",
        help="With --endpoint raw: path starting with /v2/... or /v3/...",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Stock aggs multiplier (e.g. 5 with --timespan minute)",
    )
    parser.add_argument(
        "--timespan",
        default="day",
        help="Stock aggs timespan: minute, hour, day, week, month, quarter, year",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        default="2024-01-01",
        help="Stock aggs start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default="2024-01-10",
        help="Stock aggs end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Option snapshot / trades / quotes: limit per page (defaults vary by endpoint)",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Trades/quotes: YYYY-MM-DD filter (passed as timestamp=)",
    )
    args = parser.parse_args()

    client = MassiveClient()
    ticker = str(args.ticker).upper()

    if args.endpoint == "raw":
        if not args.path.startswith("/v") and not args.path.startswith("/"):
            print(
                "error: --path should start with / (e.g. /v3/snapshot/options/SPY)", file=sys.stderr
            )
            sys.exit(2)
        path = args.path if args.path.startswith("/") else "/" + args.path
        data = client.get(path)
    elif args.endpoint == "option-snapshot":
        params = {}
        if args.limit is not None:
            params["limit"] = args.limit
        data = client.option_chain_snapshot(ticker, **params)
    elif args.endpoint == "stocks-trades":
        tparams: dict[str, str | int | float | bool | None] = {}
        if args.limit is not None:
            tparams["limit"] = args.limit
        if args.timestamp:
            tparams["timestamp"] = args.timestamp
        data = client.stock_trades(ticker, **tparams)
    elif args.endpoint == "stocks-quotes":
        qparams: dict[str, str | int | float | bool | None] = {}
        if args.limit is not None:
            qparams["limit"] = args.limit
        if args.timestamp:
            qparams["timestamp"] = args.timestamp
        data = client.stock_quotes(ticker, **qparams)
    else:
        data = client.stock_aggs_range(
            ticker,
            args.multiplier,
            args.timespan,
            args.from_date,
            args.to_date,
        )

    print(json.dumps(data, indent=2, default=str)[:50000])


if __name__ == "__main__":
    main()
