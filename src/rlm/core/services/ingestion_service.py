"""IngestionService — application layer for market data ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class IngestionResult:
    bars_path: Path
    bars_count: int
    chain_path: Path | None = None
    chain_count: int = 0


class IngestionService:
    """Fetch and normalize market data, writing results to the Parquet/CSV lake.

    Provider routing:
    - ``yfinance``: equity bars via yfinance (no credentials required)
    - ``ibkr``: equity bars via IBKR TWS (requires ibapi + running TWS)
    - ``massive``: flat-file options from Massive (requires boto3 + S3 creds)
    """

    def run(self, req: IngestionRequest) -> IngestionResult:
        log.info(
            "IngestionService.run symbol=%s source=%s interval=%s",
            req.symbol,
            req.source,
            req.interval,
        )

        if req.source == "yfinance":
            return self._fetch_yfinance(req)
        if req.source == "ibkr":
            return self._fetch_ibkr(req)
        if req.source == "massive":
            return self._fetch_massive(req)

        raise ValueError(f"Unknown ingestion source: {req.source!r}")

    def _fetch_yfinance(self, req: IngestionRequest) -> IngestionResult:
        import pandas as pd
        import yfinance as yf

        log.info("Fetching %s via yfinance (interval=%s)", req.symbol, req.interval)
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

        root = Path(__file__).resolve().parents[5]
        out_path = root / f"data/raw/bars_{req.symbol}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        log.info("Wrote %d bars → %s", len(df), out_path)

        return IngestionResult(bars_path=out_path, bars_count=len(df))

    def _fetch_ibkr(self, req: IngestionRequest) -> IngestionResult:
        from rlm.data.ibkr_stocks import fetch_ibkr_bars

        log.info("Fetching %s via IBKR", req.symbol)
        df, out_path = fetch_ibkr_bars(
            symbol=req.symbol,
            start=req.start,
            end=req.end,
            bar_size=req.interval,
        )
        return IngestionResult(bars_path=Path(out_path), bars_count=len(df))

    def _fetch_massive(self, req: IngestionRequest) -> IngestionResult:
        raise NotImplementedError(
            "Massive flat-file ingestion: use scripts/fetch_massive.py for now. "
            "Full rlm ingest --source massive support coming soon."
        )
