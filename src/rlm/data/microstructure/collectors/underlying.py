"""
Real-time underlying bar collector for the RLM microstructure layer.

Fetches 5-second OHLCV bars from Interactive Brokers via ``ib_insync`` and
persists them as date-partitioned Parquet files.  IBKR's minimum real-time bar
resolution is 5 seconds (``reqRealTimeBars``); 1-second resolution requires
tick-by-tick data which is subject to separate entitlement.

Supports two modes:
  - **snapshot**   : Pull the last N minutes of historical 5-second bars.
  - **stream**     : Continuously append real-time bars to today's Parquet file.

Usage::

    # CLI
    python -m rlm.microstructure.collectors.underlying --symbol SPY --stream

    # Programmatic
    from rlm.data.microstructure.collectors.underlying import UnderlyingCollector
    collector = UnderlyingCollector("SPY")
    collector.fetch_snapshot(minutes=10)   # Returns a DataFrame
    await collector.stream()               # Runs until cancelled
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rlm.data.ibkr_stocks import ibkr_bars_to_dataframe, load_ibkr_socket_config

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Stock

    _HAS_IB_INSYNC = True
except ImportError:
    _HAS_IB_INSYNC = False

_SCHEMA_COLS = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]


def _microstructure_underlying_path(symbol: str, date: "datetime.date | str", data_root: str) -> Path:
    return Path(data_root) / f"underlying/{symbol}/1s/{symbol}_{date}.parquet"


def _append_or_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Merge new bars with any existing file, dedup by timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------


class UnderlyingCollector:
    """
    Fetches and persists real-time underlying bars from Interactive Brokers.

    Parameters
    ----------
    symbol      : Ticker symbol (e.g. "SPY")
    data_root   : Root of the microstructure data lake
    bar_size    : Bar size string for historical data (e.g. "5 secs", "1 min")
    client_id   : IBKR client ID (uses env var ``IBKR_CLIENT_ID`` by default)
    """

    def __init__(
        self,
        symbol: str,
        *,
        data_root: str = "data/microstructure",
        bar_size: str = "5 secs",
        client_id: int | None = None,
    ) -> None:
        if not _HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is required for UnderlyingCollector.\n"
                "Install with: pip install 'regime-locus-matrix[ib-insync]'"
            )
        self.symbol = symbol.upper()
        self.data_root = data_root
        self.bar_size = bar_size
        self._host, self._port, self._cid = load_ibkr_socket_config()
        if client_id is not None:
            self._cid = client_id
        self._ib: IB | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> IB:
        ib = IB()
        ib.connect(self._host, self._port, clientId=self._cid, timeout=30)
        logger.info("Connected to IBKR at %s:%s (cid=%s)", self._host, self._port, self._cid)
        return ib

    def _stock_contract(self) -> Any:
        return Stock(self.symbol, "SMART", "USD")

    # ------------------------------------------------------------------
    # Snapshot (historical) fetch
    # ------------------------------------------------------------------

    def fetch_snapshot(self, *, minutes: int = 60) -> pd.DataFrame:
        """
        Pull the last *minutes* of 5-second bars.

        Returns
        -------
        DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        Saves the result to the appropriate date-partitioned Parquet file.
        """
        ib = self._connect()
        try:
            contract = self._stock_contract()
            duration = f"{max(minutes * 60, 300)} S"
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=self.bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
        finally:
            ib.disconnect()

        if not bars:
            logger.warning("No bars returned for %s", self.symbol)
            return pd.DataFrame(columns=_SCHEMA_COLS)

        df = ibkr_bars_to_dataframe(bars)
        df = df[_SCHEMA_COLS].sort_values("timestamp").reset_index(drop=True)
        self._save(df)
        logger.info("Snapshot: %d bars for %s", len(df), self.symbol)
        return df

    # ------------------------------------------------------------------
    # Real-time streaming
    # ------------------------------------------------------------------

    async def stream(self, *, poll_seconds: float = 5.0, max_runtime_hours: float = 8.0) -> None:
        """
        Stream real-time 5-second bars, appending to today's Parquet file.

        Parameters
        ----------
        poll_seconds        : How often to flush accumulated bars (seconds)
        max_runtime_hours   : Hard stop after this many hours (for safety)
        """
        ib = IB()
        ib.connect(self._host, self._port, clientId=self._cid + 10)
        contract = self._stock_contract()
        buffer: list[dict[str, Any]] = []

        def _on_bar(bars: Any, has_new_bar: bool) -> None:
            if not has_new_bar or not bars:
                return
            bar = bars[-1]
            buffer.append(
                {
                    "timestamp": pd.Timestamp(bar.time),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.wap) if bar.wap and bar.wap > 0 else float("nan"),
                }
            )

        # 5-second real-time bars
        rt_bars = ib.reqRealTimeBars(contract, 5, "TRADES", useRTH=True)
        rt_bars.updateEvent += _on_bar

        logger.info("Streaming %s real-time bars (Ctrl-C to stop)", self.symbol)
        deadline = time.monotonic() + max_runtime_hours * 3600

        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(poll_seconds)
                if buffer:
                    batch = pd.DataFrame(buffer.copy())
                    buffer.clear()
                    self._save(batch)
                    logger.debug("Flushed %d bars", len(batch))
        except asyncio.CancelledError:
            pass
        finally:
            ib.cancelRealTimeBars(rt_bars)
            ib.disconnect()
            logger.info("Stream stopped for %s", self.symbol)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame) -> None:
        """Split by date and append to date-partitioned Parquet."""
        df = df.copy()
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
        for date, group in df.groupby("_date"):
            path = _microstructure_underlying_path(self.symbol, date, self.data_root)
            _append_or_write_parquet(group.drop(columns="_date"), path)
            logger.debug("Wrote %d rows → %s", len(group), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch real-time underlying bars from IBKR into the microstructure lake."
    )
    parser.add_argument("--symbol", required=True, help="Ticker (e.g. SPY)")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "stream"],
        default="snapshot",
        help="'snapshot' fetches recent history; 'stream' runs continuously (default: snapshot)",
    )
    parser.add_argument("--minutes", type=int, default=60, help="Minutes of history for snapshot mode")
    parser.add_argument("--data-root", default="data/microstructure", help="Root of microstructure lake")
    parser.add_argument("--bar-size", default="5 secs", help="IBKR bar size string")
    args = parser.parse_args()

    collector = UnderlyingCollector(args.symbol, data_root=args.data_root, bar_size=args.bar_size)

    if args.mode == "snapshot":
        df = collector.fetch_snapshot(minutes=args.minutes)
        print(f"Fetched {len(df)} bars for {args.symbol}")
        if not df.empty:
            print(df.tail(5).to_string())
    else:
        asyncio.run(collector.stream())


if __name__ == "__main__":
    _main()
