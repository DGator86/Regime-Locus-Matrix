"""IngestionService — application layer for market data ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rlm.core.run_manifest import RunManifest, new_run_id, now_utc
from rlm.data.paths import get_raw_data_dir
from rlm.utils.logging import get_logger
from rlm.utils.timing import timed_stage

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
    data_root: str | None = None
    backend: str = "auto"
    write_lake: bool = False
    run_id: str = field(default_factory=new_run_id)


@dataclass
class IngestionArtifacts:
    """Paths and metadata for files written by IngestionService."""

    bars_path: Path | None = None
    bars_count: int = 0
    chain_path: Path | None = None
    chain_count: int = 0
    duration_s: float = 0.0


class IngestionService:
    """Fetch and normalize market data, writing results to the raw/ directory.

    Provider routing:
    - ``yfinance``: equity bars via yfinance (no credentials required)
    - ``ibkr``: equity bars via IBKR TWS (requires ibapi + running TWS)
    - ``massive``: flat-file options from Massive (requires boto3 + S3 creds)

    Data root resolution is delegated to ``rlm.data.paths.get_raw_data_dir``.
    """

    def run(self, req: IngestionRequest) -> IngestionArtifacts:
        raw_dir = get_raw_data_dir(req.data_root)
        raw_dir.mkdir(parents=True, exist_ok=True)

        with timed_stage(
            log, "ingest",
            run_id=req.run_id, symbol=req.symbol,
            source=req.source, interval=req.interval,
        ):
            if req.source == "yfinance":
                arts = self._fetch_yfinance(req, raw_dir)
            elif req.source == "ibkr":
                arts = self._fetch_ibkr(req, raw_dir)
            elif req.source == "massive":
                arts = self._fetch_massive(req, raw_dir)
            else:
                raise ValueError(f"Unknown ingestion source: {req.source!r}")

        self._write_manifest(req, arts)
        return arts

    def write_outputs(self, req: IngestionRequest, arts: IngestionArtifacts) -> IngestionArtifacts:
        """Optionally write Parquet lake artifacts after CSV ingest.

        When ``req.write_lake=True`` and pyarrow is available, converts the
        freshly written CSV into Parquet lake format.  No-op otherwise.
        """
        if not req.write_lake or arts.bars_path is None or not arts.bars_path.is_file():
            return arts

        try:
            from rlm.data.lake.writers import write_bars_parquet
            df = pd.read_csv(arts.bars_path, parse_dates=["timestamp"])
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            write_bars_parquet(df, req.symbol, interval=req.interval, data_root=req.data_root)
            log.info(
                "ingest lake write  symbol=%s rows=%d interval=%s",
                req.symbol, len(df), req.interval,
            )
        except Exception as exc:
            log.warning("ingest lake write failed  symbol=%s error=%s", req.symbol, exc)

        return arts

    def summarize(self, arts: IngestionArtifacts) -> dict:
        """Return a human-readable summary dict from the ingestion artifacts."""
        return {
            "bars_count": arts.bars_count,
            "chain_count": arts.chain_count,
            "bars_path": str(arts.bars_path) if arts.bars_path else None,
            "chain_path": str(arts.chain_path) if arts.chain_path else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, req: IngestionRequest, raw_dir: Path) -> IngestionArtifacts:
        import yfinance as yf

        log.info(
            "yfinance fetch  symbol=%s interval=%s start=%s end=%s",
            req.symbol, req.interval, req.start, req.end,
        )
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

        bars_path = raw_dir / f"bars_{req.symbol}.csv"
        df.to_csv(bars_path)
        log.info("yfinance bars written  rows=%d path=%s", len(df), bars_path)

        arts = IngestionArtifacts(bars_path=bars_path, bars_count=len(df))

        if req.fetch_options:
            arts = self._fetch_yfinance_options(req, raw_dir, arts)

        return arts

    def _fetch_yfinance_options(
        self, req: IngestionRequest, raw_dir: Path, arts: IngestionArtifacts
    ) -> IngestionArtifacts:
        import yfinance as yf

        try:
            ticker = yf.Ticker(req.symbol)
            expiries = ticker.options
            if not expiries:
                log.warning("yfinance no option expiries  symbol=%s", req.symbol)
                return arts

            frames = []
            for exp in expiries[:4]:  # limit to nearest 4 expiries
                try:
                    chain = ticker.option_chain(exp)
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    calls["expiry"] = exp
                    puts["expiry"] = exp
                    calls["option_type"] = "call"
                    puts["option_type"] = "put"
                    frames.extend([calls, puts])
                except Exception as exc:
                    log.debug("yfinance option chain error  exp=%s error=%s", exp, exc)

            if frames:
                chain_df = pd.concat(frames, ignore_index=True)
                chain_path = raw_dir / f"option_chain_{req.symbol}.csv"
                chain_df.to_csv(chain_path, index=False)
                log.info("yfinance chain written  rows=%d path=%s", len(chain_df), chain_path)
                arts.chain_path = chain_path
                arts.chain_count = len(chain_df)
        except Exception as exc:
            log.warning("yfinance options failed  symbol=%s error=%s", req.symbol, exc)

        return arts

    def _fetch_ibkr(self, req: IngestionRequest, raw_dir: Path) -> IngestionArtifacts:
        from rlm.data.ibkr_stocks import fetch_ibkr_bars

        log.info("ibkr fetch  symbol=%s", req.symbol)
        df, out_path_str = fetch_ibkr_bars(
            symbol=req.symbol,
            start=req.start,
            end=req.end,
            bar_size=req.interval,
        )
        return IngestionArtifacts(bars_path=Path(out_path_str), bars_count=len(df))

    def _fetch_massive(self, req: IngestionRequest, raw_dir: Path) -> IngestionArtifacts:
        raise NotImplementedError(
            "Massive flat-file ingestion: use scripts/fetch_massive.py for now.\n"
            "Full rlm ingest --source massive support coming in a future release."
        )

    def _write_manifest(self, req: IngestionRequest, arts: IngestionArtifacts) -> None:
        manifest = RunManifest(
            run_id=req.run_id,
            command="ingest",
            symbol=req.symbol,
            timestamp_utc=now_utc(),
            config_summary={
                "source": req.source,
                "interval": req.interval,
                "start": req.start,
                "end": req.end,
                "fetch_options": req.fetch_options,
            },
            output_paths={
                "bars_csv": str(arts.bars_path) if arts.bars_path else "",
                "chain_csv": str(arts.chain_path) if arts.chain_path else "",
            },
            metrics=self.summarize(arts),
            backend=req.backend,
            success=True,
            duration_s=arts.duration_s,
        )
        manifest.write(req.data_root)
