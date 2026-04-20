"""IngestionService — application layer for market data ingestion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rlm.data.paths import get_artifacts_dir, get_raw_data_dir
from rlm.data.providers import IBKRProvider, MarketDataProvider, YFinanceProvider
from rlm.utils.run_id import generate_run_id


@dataclass
class IngestionRequest:
    symbol: str
    source: str = "yfinance"
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    fetch_options: bool = False
    data_root: str | None = None
    backend: str = "auto"
    profile: str | None = None
    config_path: str | None = None
    write_manifest: bool = True


@dataclass
class IngestionResult:
    bars_path: Path
    bars_count: int
    chain_path: Path | None = None
    chain_count: int = 0
    duration_s: float = 0.0
    backend: str = "csv"
    provider: str = ""
    run_id: str | None = None
    manifest_path: Path | None = None
    metadata_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionService:
    _PROVIDERS: dict[str, type[MarketDataProvider]] = {
        "yfinance": YFinanceProvider,
        "ibkr": IBKRProvider,
    }

    def run(self, req: IngestionRequest) -> IngestionResult:
        t0 = time.monotonic()
        run_id = generate_run_id("ingest")
        provider = self.resolve_provider(req.source)
        backend = self._resolve_backend(req.backend)

        bars = provider.fetch_bars(symbol=req.symbol, start=req.start, end=req.end, interval=req.interval)
        bars_path = self._write_frame(bars.bars_df, kind="bars", symbol=req.symbol, data_root=req.data_root, backend=backend)

        chain_path: Path | None = None
        chain_count = 0
        chain_meta: dict[str, Any] = {}
        if req.fetch_options:
            chain = provider.fetch_option_chain(symbol=req.symbol)
            chain_meta = chain.metadata
            if chain.chain_df is not None and not chain.chain_df.empty:
                chain_count = len(chain.chain_df)
                chain_path = self._write_frame(chain.chain_df, kind="chain", symbol=req.symbol, data_root=req.data_root, backend=backend)

        metadata = {
            "source": req.source,
            "backend": backend,
            "symbol": req.symbol,
            "bars": bars.metadata,
            "chain": chain_meta,
            "profile": req.profile,
            "config_path": req.config_path,
        }

        metadata_path: Path | None = None
        manifest_path: Path | None = None
        if req.write_manifest:
            metadata_path = self._write_ingest_metadata(req, run_id, metadata, bars_path, chain_path)
            manifest_path = self._write_manifest(req, run_id, bars_path, chain_path, metadata_path)

        return IngestionResult(
            bars_path=bars_path,
            bars_count=len(bars.bars_df),
            chain_path=chain_path,
            chain_count=chain_count,
            duration_s=time.monotonic() - t0,
            backend=backend,
            provider=req.source,
            run_id=run_id,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def resolve_provider(self, source: str) -> MarketDataProvider:
        key = source.lower().strip()
        if key not in self._PROVIDERS:
            raise ValueError(f"Unknown ingestion source: {source!r}")
        return self._PROVIDERS[key]()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        key = backend.lower().strip()
        if key not in {"auto", "csv", "lake"}:
            raise ValueError(f"Unsupported backend: {backend!r}")
        return "lake" if key == "auto" else key

    def _write_frame(self, df: pd.DataFrame, *, kind: str, symbol: str, data_root: str | None, backend: str) -> Path:
        raw_dir = get_raw_data_dir(data_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        sym = symbol.upper()

        if backend == "lake":
            return self._write_lake(df, raw_dir, kind, sym)
        if backend == "csv":
            return self._write_csv(df, raw_dir, kind, sym)
        raise ValueError(f"Unsupported backend: {backend!r}")

    @staticmethod
    def _write_lake(df: pd.DataFrame, raw_dir: Path, kind: str, symbol: str) -> Path:
        out_path = raw_dir / "lake" / ("bars" if kind == "bars" else "chains") / f"{symbol}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path

    @staticmethod
    def _write_csv(df: pd.DataFrame, raw_dir: Path, kind: str, symbol: str) -> Path:
        out_path = raw_dir / (f"bars_{symbol}.csv" if kind == "bars" else f"option_chain_{symbol}.csv")
        df.to_csv(out_path, index=False)
        return out_path

    def _write_ingest_metadata(
        self,
        req: IngestionRequest,
        run_id: str,
        metadata: dict[str, Any],
        bars_path: Path,
        chain_path: Path | None,
    ) -> Path:
        artifact_dir = get_artifacts_dir(req.data_root) / "ingest" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out = artifact_dir / "metadata.json"
        out.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "symbol": req.symbol,
                    "source": req.source,
                    "backend": req.backend,
                    "bars_path": str(bars_path),
                    "chain_path": str(chain_path) if chain_path else None,
                    "metadata": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out

    def _write_manifest(
        self,
        req: IngestionRequest,
        run_id: str,
        bars_path: Path,
        chain_path: Path | None,
        metadata_path: Path | None,
    ) -> Path:
        artifact_dir = get_artifacts_dir(req.data_root) / "ingest" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        out = artifact_dir / "run_manifest.json"
        out.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "command": "ingest",
                    "symbol": req.symbol,
                    "source": req.source,
                    "backend": req.backend,
                    "artifacts": {
                        "bars_path": str(bars_path),
                        "chain_path": str(chain_path) if chain_path else None,
                        "metadata_path": str(metadata_path) if metadata_path else None,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out
