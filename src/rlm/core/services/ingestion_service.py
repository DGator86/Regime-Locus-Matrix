"""IngestionService — application layer for market data ingestion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from rlm.data.paths import get_artifacts_dir, get_raw_data_dir
from rlm.data.providers import IBKRProvider, MassiveProvider, MarketDataProvider, YFinanceProvider
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
    """Fetch and normalize market data via provider adapters and backend writers."""

    _PROVIDERS: dict[str, type[MarketDataProvider]] = {
        "yfinance": YFinanceProvider,
        "ibkr": IBKRProvider,
        "massive": MassiveProvider,
    }

    def run(self, req: IngestionRequest) -> IngestionResult:
        t0 = time.monotonic()
        run_id = self._new_run_id(req.symbol)
        provider = self._select_provider(req.source)
        backend = self._resolve_backend(req.backend)

        log.info(
            "ingest start symbol=%s source=%s backend=%s interval=%s",
            req.symbol,
            req.source,
            backend,
            req.interval,
        )

        bars = provider.fetch_bars(
            symbol=req.symbol,
            start=req.start,
            end=req.end,
            interval=req.interval,
        )
        bars_path = self._write_frame(
            bars.bars_df,
            kind="bars",
            symbol=req.symbol,
            data_root=req.data_root,
            backend=backend,
        )

        chain_path: Path | None = None
        chain_count = 0
        chain_meta: dict[str, Any] = {}
        if req.fetch_options:
            chain = provider.fetch_option_chain(symbol=req.symbol)
            chain_meta = chain.metadata
            if chain.chain_df is not None and not chain.chain_df.empty:
                chain_count = len(chain.chain_df)
                chain_path = self._write_frame(
                    chain.chain_df,
                    kind="chain",
                    symbol=req.symbol,
                    data_root=req.data_root,
                    backend=backend,
                )

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

        result = IngestionResult(
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
        log.info("ingest done symbol=%s bars=%d backend=%s", req.symbol, result.bars_count, backend)
        return result

    def _select_provider(self, source: str) -> MarketDataProvider:
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

    def _write_frame(
        self,
        df,
        *,
        kind: str,
        symbol: str,
        data_root: str | None,
        backend: str,
    ) -> Path:
        raw_dir = get_raw_data_dir(data_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        sym = symbol.upper()

        if backend == "lake":
            if kind == "bars":
                out_path = raw_dir / "lake" / "bars" / f"{sym}.parquet"
            else:
                out_path = raw_dir / "lake" / "chains" / f"{sym}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            return out_path

        out_path = raw_dir / (f"bars_{sym}.csv" if kind == "bars" else f"option_chain_{sym}.csv")
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
        payload = {
            "run_id": run_id,
            "symbol": req.symbol,
            "source": req.source,
            "backend": req.backend,
            "bars_path": str(bars_path),
            "chain_path": str(chain_path) if chain_path else None,
            "metadata": metadata,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
        out = artifact_dir / "manifest.json"
        payload = {
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
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def _new_run_id(symbol: str) -> str:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return f"{ts}_{symbol.upper()}_{uuid4().hex[:8]}"
