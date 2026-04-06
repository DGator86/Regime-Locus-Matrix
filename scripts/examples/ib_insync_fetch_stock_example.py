"""
Standalone example: stock bars via ib_insync (not used by RLM core — we use ibapi).

Install: pip install ib_insync pyarrow
Set IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID or edit constants below.

Run from repo root::

    python scripts/examples/ib_insync_fetch_stock_example.py
"""

from __future__ import annotations

import os
from pathlib import Path

from ib_insync import IB, Stock, util

HOST = os.environ.get("IBKR_HOST", "127.0.0.1")
PORT = int(os.environ.get("IBKR_PORT", "7497"))
CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "1"))


def fetch_stock(symbol: str, duration: str, bar_size: str, outpath: str) -> None:
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    contract = Stock(symbol, "SMART", "USD")

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    df = util.df(bars)
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

    ib.disconnect()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    fetch_stock("SPY", "2 Y", "1 day", str(root / "data/stocks/SPY/1d/spy_2y_1d.parquet"))
    fetch_stock("SPY", "10 D", "1 min", str(root / "data/stocks/SPY/1m/spy_10d_1m.parquet"))
