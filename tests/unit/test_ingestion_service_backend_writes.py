from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlm.core.services.ingestion_service import IngestionRequest, IngestionService


class _StubProvider:
    def fetch_bars(self, *, symbol: str, start: str | None, end: str | None, interval: str):
        del symbol, start, end, interval
        from rlm.data.providers.base import ProviderBarsResult

        return ProviderBarsResult(
            pd.DataFrame([{"timestamp": "2025-01-01", "open": 1.0}]), source="stub"
        )

    def fetch_option_chain(self, *, symbol: str):
        del symbol
        from rlm.data.providers.base import ProviderOptionChainResult

        return ProviderOptionChainResult(None, source="stub")


def test_ingestion_csv_and_lake_writes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rlm.core.services.ingestion_service.resolve_provider", lambda _name: _StubProvider()
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: Path(path).write_text("ok", encoding="utf-8"),
    )

    csv_req = IngestionRequest(symbol="SPY", source="stub", data_root=str(tmp_path), backend="csv")
    csv_result = IngestionService().run(csv_req)
    assert csv_result.bars_path.suffix == ".csv"

    lake_req = IngestionRequest(
        symbol="SPY", source="stub", data_root=str(tmp_path), backend="lake"
    )
    lake_result = IngestionService().run(lake_req)
    assert lake_result.bars_path.suffix == ".parquet"
