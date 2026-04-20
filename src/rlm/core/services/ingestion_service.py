"""IngestionService — application layer for market data ingestion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rlm.data.paths import get_raw_data_dir
from rlm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IngestionRequest:
    """Input bundle for a data ingestion run."""

    symbol: str
    source: str = "yfinance"
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    fetch_options: bool = False
    data_root: str | None = None  # maps to --data-root / RLM_DATA_ROOT


@dataclass
class IngestionResult:
    bars_path: Path
    bars_count: int
    chain_path: Path | None = None
    chain_count: int = 0
    duration_s: float = 0.0


class IngestionService:
    """Fetch and normalize market data, writing results to the Parquet/CSV lake.

    Provider routing:
    - ``yfinance``: equity bars via yfinance (no credentials required)
    - ``ibkr``: equity bars via IBKR TWS (requires ibapi + running TWS)
    - ``massive``: flat-file options from Massive (requires boto3 + S3 creds)

    Data root resolution is delegated to ``rlm.data.paths.get_raw_data_dir``.
    """

    def run(self, req: IngestionRequest) -> IngestionResult:
        t0 = time.monotonic()
        raw_dir = get_raw_data_dir(req.data_root)
        raw_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "ingest start  symbol=%s source=%s interval=%s dest=%s",
            req.symbol, req.source, req.interval, raw_dir,
        )

        if req.source == "yfinance":
            result = self._fetch_yfinance(req, raw_dir)
        elif req.source == "ibkr":
            result = self._fetch_ibkr(req, raw_dir)
        elif req.source == "massive":
            result = self._fetch_massive(req, raw_dir)
        else:
            raise ValueError(f"Unknown ingestion source: {req.source!r}")

        result.duration_s = time.monotonic() - t0
        log.info(
            "ingest done  symbol=%s bars=%d path=%s duration=%.1fs",
            req.symbol, result.bars_count, result.bars_path, result.duration_s,
        )
        return result

    def _fetch_yfinance(self, req: IngestionRequest, raw_dir: Path) -> IngestionResult:
        import pandas as pd
        import yfinance as yf

        log.info("yfinance fetch  symbol=%s interval=%s start=%s end=%s", req.symbol, req.interval, req.start, req.end)
        ticker = yf.Ticker(req.symbol)
        df = ticker.history(
            start=req.start,
            end=req.end,
            interval=req.interval,
            auto_adjust=True,
        )
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {req.symbol}")

        df.index.name = "timestamp"
        df.columns = [c.lower() for c in df.columns]

        out_path = raw_dir / f"bars_{req.symbol}.csv"
        df.to_csv(out_path)
        log.info("yfinance wrote  rows=%d path=%s", len(df), out_path)

        return IngestionResult(bars_path=out_path, bars_count=len(df))

    def _fetch_ibkr(self, req: IngestionRequest, raw_dir: Path) -> IngestionResult:
        from rlm.data.ibkr_stocks import fetch_ibkr_bars

        log.info("ibkr fetch  symbol=%s", req.symbol)
        df, out_path_str = fetch_ibkr_bars(
            symbol=req.symbol,
            start=req.start,
            end=req.end,
            bar_size=req.interval,
        )
        return IngestionResult(bars_path=Path(out_path_str), bars_count=len(df))

    def _fetch_massive(self, req: IngestionRequest, raw_dir: Path) -> IngestionResult:
        raise NotImplementedError(
            "Massive flat-file ingestion: use scripts/fetch_massive.py for now.\n"
            "Full rlm ingest --source massive support coming in a future release."
        )
