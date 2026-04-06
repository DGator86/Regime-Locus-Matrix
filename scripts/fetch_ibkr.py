"""CLI smoke test for IBKR historical stock bars (requires TWS/Gateway + pip install '.[ibkr]')."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical stock bars from Interactive Brokers")
    parser.add_argument("ticker", nargs="?", default="SPY", help="US stock symbol")
    parser.add_argument("--duration", default="5 D", help='IB duration string, e.g. "5 D", "1 M"')
    parser.add_argument("--bar-size", default="1 day", help='IB bar size, e.g. "1 day", "1 hour"')
    parser.add_argument("--what", default="TRADES", dest="what_to_show", help="TRADES, MIDPOINT, etc.")
    parser.add_argument("--end", default="", dest="end_datetime", help='End datetime (IB format); empty = now')
    parser.add_argument("--rth", type=int, default=1, choices=(0, 1), help="1 = regular hours only")
    parser.add_argument("--host", default=None, help="Override IBKR_HOST")
    parser.add_argument("--port", type=int, default=None, help="Override IBKR_PORT")
    parser.add_argument("--client-id", type=int, default=None, help="Override IBKR_CLIENT_ID")
    args = parser.parse_args()

    try:
        from rlm.data.ibkr_stocks import fetch_historical_stock_bars
    except ImportError as e:
        print("error: install IB API support: pip install -e '.[ibkr]'", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(2)

    df = fetch_historical_stock_bars(
        str(args.ticker).upper(),
        duration=args.duration,
        bar_size=args.bar_size,
        what_to_show=args.what_to_show,
        use_rth=args.rth,
        end_datetime=args.end_datetime,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
